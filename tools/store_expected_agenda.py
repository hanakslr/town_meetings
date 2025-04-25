import json
from typing import Any, Dict
from pathlib import Path

from anthropic.types import ToolParam

from tools import Tool


class StoreExpectedAgendas(Tool):
    """A tool class for populating the expected output for committees and their fetching strategy"""

    name = "store_expected_agendas"

    @classmethod
    def get_tool_definition(cls) -> ToolParam:
        return {
            "name": StoreExpectedAgendas.name,
            "description": f"""Store all the agendas for a certain committee. 
            These will be saved into a fixture file that will be used as the expected results for this committee.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The committees name"},
                    "meetings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {
                                    "type": "string",
                                    "description": "Date of the meeting in YYYY-MM-DD format",
                                },
                                "agenda": {
                                    "type": "string",
                                    "description": "Link to the agenda itself. This could be a URL that show the agenda in HTML or a link to a PDF/other file.",
                                },
                            },
                        },
                    },
                },
            },
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # TODO probably put some assertions here and check the data
        name = params.get("name")
        meetings = params.get("meetings")

        expected_output_dir = Path("strategies/expected")

        file_path = expected_output_dir / f"{name.replace(' ', '_').lower()}.json"

        with open(file_path, "w") as f:
            json.dump(meetings, f, indent=4)
