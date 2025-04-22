from anthropic.types import ToolParam

from tools import Tool


class PytestTool(Tool):
    name = "unit_test_code"

    @classmethod
    def get_tool_definition(cls) -> ToolParam:
        return {
            "name": PytestTool.name,
            "description": "Given execution code, a pytest unit test",
        }
