"""
Functionality and helper classes (CST Transformers) to dump a json strategy and code
snippet into a useable class.
"""
import libcst as cst
from libcst import (
    Arg,
    ClassDef,
    FunctionDef,
    IndentedBlock,
    Param,
    Parameters,
    parse_module,
    ImportAlias,
    Name,
    ImportFrom,
    Attribute,
    SimpleStatementLine
)
from pathlib import Path
from typing import Any

class WrapFunctionInClassWithAttribute(cst.CSTTransformer):
    """
    Given a function name, wrap that function in a new class.
    This is a very specific implementation to the strategy dumping use case.
    """
    def __init__(
        self,
        target_func: str, 
        class_name: str,
        attr_name: str,
        attr_value: str,
        class_doc: str,
        method_doc: str,
    ):
        """
        Args
            target_func: name of the function we want to wrap
            class_name: name of the class we want to create
            attr_name: name of an attribute to add to the class
            attr_value: value of the attribute for the class
            class_doc: docustring for the class
            method_doc: docustring for the target_func
        """
        self.target_func = target_func
        self.class_name = class_name
        self.attr_name = attr_name
        self.attr_value = attr_value
        self.class_doc = class_doc
        self.method_doc = method_doc
        self.found_func: cst.FunctionDef | None = None

    def leave_FunctionDef(
        self, original_node: FunctionDef, updated_node: FunctionDef
    ) -> cst.RemovalSentinel | None:
        if original_node.name.value == self.target_func:
            self.found_func = updated_node
            return cst.RemoveFromParent()
        return updated_node

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        if not self.found_func:
            return updated_node

        method_docstring = SimpleStatementLine(
            body=[cst.Expr(value=cst.SimpleString(f'"""{self.method_doc}"""'))]
        )

        # Add `self` param
        new_params = [Param(Name("self"))] + list(self.found_func.params.params)
        method_with_self = self.found_func.with_changes(
            params=Parameters(params=new_params),
            body=IndentedBlock(
                body=[method_docstring] + list(self.found_func.body.body)
            ),
        )

        # Create class-level assignment
        assign = SimpleStatementLine(
            body=[
                cst.Assign(
                    targets=[cst.AssignTarget(target=Name(self.attr_name))],
                    value=cst.SimpleString(f'"{self.attr_value}"'),
                )
            ]
        )

        class_docstring = SimpleStatementLine(
            body=[cst.Expr(value=cst.SimpleString(f'"""{self.class_doc}"""'))]
        )

        new_class = ClassDef(
            name=Name(self.class_name),
            bases=[Arg(value=Name("FetchingStrategy"))],
            body=IndentedBlock(body=[class_docstring, assign, method_with_self]),
        )

        # Insert the class after the last import
        new_body = []
        inserted = False
        for stmt in updated_node.body:
            if not inserted and not isinstance(
                stmt, (cst.SimpleStatementLine, cst.ImportFrom, cst.Import)
            ):
                new_body.append(new_class)
                inserted = True
            new_body.append(stmt)

        if not inserted:
            new_body.append(new_class)

        return updated_node.with_changes(body=new_body)

class AddFromImportTransformer(cst.CSTTransformer):
    def __init__(self, module: str, name: str, alias: str = None):
        self.module = module
        self.name = name
        self.alias = alias
        self.import_already_exists = False

    def leave_ImportFrom(self, original_node, updated_node):
        if isinstance(original_node.module, (Name, Attribute)):
            modname = (
                original_node.module.attr.value
                if isinstance(original_node.module, Attribute)
                else original_node.module.value
            )
            if modname == self.module:
                for alias in original_node.names:
                    if isinstance(alias, ImportAlias):
                        if alias.name.value == self.name:
                            self.import_already_exists = True
        return updated_node

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        if self.import_already_exists:
            return updated_node

        new_import = ImportFrom(
            module=Name(self.module),
            names=[
                ImportAlias(
                    name=Name(self.name),
                    asname=cst.AsName(name=Name(self.alias))
                    if self.alias
                    else None,
                )
            ],
            relative=[],
        )
        new_stmt = SimpleStatementLine(body=[new_import])
        return updated_node.with_changes(body=[new_stmt] + list(updated_node.body))
    


def save_fetching_strategy(fetch_result: dict[str, Any]):
    """
    Given the fetch_result, create a class that is a subclass of FetchingStrategy
    that uses the code and schema that the fetch result provides.
    """
    # Store the fetching strategy to a file.
    strategies_dir = Path("strategies")
    strategy_name = fetch_result.get("strategy_name")
    file_path = strategies_dir / f"{strategy_name}.py"

    original_code = fetch_result.get("code")
    module = parse_module(original_code)

    schema: dict[str, str] = fetch_result.get("schema")
    arg_descriptions = "\n            ".join([f"{k}: {v}" for k, v in schema.items()])

    method_doc = f"""
        Args:
            {arg_descriptions}
        
        Returns:
            List of {{date, agenda}}
    """

    # Add comment to top of file
    claude_comment = cst.EmptyLine(
        comment=cst.Comment("# This strategy was initially autogenerated by Claude.")
    )
    transform_comment = cst.EmptyLine(
        comment=cst.Comment(
            "# Additional transformation was applied afterwards to structure it as a useful class."
        )
    )
    empty_line = cst.EmptyLine()
    module = module.with_changes(
        header=[claude_comment, transform_comment, empty_line] + list(module.header)
    )

    # Apply the transformer
    import_transformer = AddFromImportTransformer("strategies", "FetchingStrategy")
    modified_module = module.visit(import_transformer)
    class_transformer = WrapFunctionInClassWithAttribute(
        target_func="get_committee_agendas",
        class_name=f"{strategy_name.replace('_', ' ').title().replace(' ', '')}",
        attr_name="name",
        attr_value=strategy_name,
        class_doc=fetch_result.get("notes", None),
        method_doc=method_doc,
    )
    modified_module = modified_module.visit(class_transformer)

    # Write the modified code to file
    with open(file_path, "w") as f:
        f.write(modified_module.code)