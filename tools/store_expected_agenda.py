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
            "description": f"""Store all the agendas for a certain committee from Oct 2024-March 2025. 
            These will be saved into a fixture file that will be used as the expected results for this committee.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The committees name"
                    },
                    "meetings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {
                                    "type": "string",
                                    "description": "Date of the meeting in YYYY-MM-DD format"
                                },
                                "agenda": {
                                    "type": "string",
                                    "description": "Link to the agenda itself."
                                }
                            }
                        }

                    }
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # TODO probably put some assertions here and check the data
        name = params.get("name")

        expected_output_dir = Path("strategies/fixtures/expected")

        file_path = expected_output_dir / f"{name.replace(" ", "-").lower()}.json"

        with open(file_path, "w") as f:
            json.dump(params, f, indent=4)



