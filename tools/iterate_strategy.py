import json
from typing import Any, Dict
from pathlib import Path
import sys
import traceback
import contextlib
import io
import ast
import types
from deepdiff import DeepDiff

from anthropic.types import ToolParam

from tools import Tool


class TestProposedStrategyTool(Tool):
    name = "test_proposed_strategy"

    @classmethod
    def get_tool_definition(cls) -> ToolParam:
        from tools.store_expected_agenda import StoreExpectedAgendas

        return {
            "name": TestProposedStrategyTool.name,
            "description": f"Test a proposed scraping strategy by comparing the input to the expected output, previously stored by the {StoreExpectedAgendas.name}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "committee_name": {
                        "type": "string",
                        "description": "Name of the committee being tested",
                    },
                    "strategy_name": {
                        "type": "string",
                        "description": "A concise descriptor of the strategy, in snake case. Example: embedded_html_links, filtered_table",
                    },
                    "schema": {
                        "type": "object",
                        "description": "Map of field names to descriptions",
                        "additionalProperties": {"type": "string"},
                    },
                    "values": {
                        "type": "object",
                        "description": "Map of schema field names to values for this specific committee",
                        "additionalProperties": True,
                    },
                    "code": {
                        "type": "string",
                        "description": "A code snippet that consumes the schema vals. The code snippet should be specific to the strategy but not contain any hard-coded references to this particular committee",
                    },
                },
                "required": [
                    "committee_name",
                    "strategy_name",
                    "schema",
                    "values",
                    "code",
                ],
            },
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        code = params.get("code")
        values = params.get("values")
        committee_name = params.get("committee_name")

        expected_output_dir = Path("strategies/fixtures/expected")
        file_path = (
            expected_output_dir / f"{committee_name.replace(' ', '_').lower()}.json"
        )
        if file_path.exists():
            with open(file_path) as f:
                expected_output = json.load(f)

        if not expected_output:
            raise Exception(f"No expected output found at {file_path}")

        return await run_test_case(code, values, expected_output)


async def run_test_case(code: str, values: dict, expected_output: list) -> dict:
    """
    Executes user-provided scraping code safely, injecting values and comparing result to expected output.
    Returns pass/fail result, logs, exception (if any), and diff.
    """
    namespace = {}
    stdout = io.StringIO()
    stderr = io.StringIO()
    output = None
    exc = None

    # Step 1: Validate the code parses as Python
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {
            "passed": False,
            "logs": "",
            "exception": f"SyntaxError: {e}",
            "diff": {},
            "actual_output": None,
        }

    # Step 2: Execute the code safely and call the function
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, namespace)
            func = next(
                (v for v in namespace.values() if isinstance(v, types.FunctionType)),
                None,
            )
            if not func:
                raise RuntimeError("No function found in code snippet")

            output = func(**values)
    except Exception as e:
        exc = traceback.format_exc()

    # Step 3: Compare result
    diff = (
        {} if output is None else DeepDiff(output, expected_output, ignore_order=True)
    )
    passed = not bool(diff) and exc is None

    return {
        "passed": passed,
        "logs": stdout.getvalue(),
        "exception": exc,
        "diff": diff,
        "actual_output": output,
    }
