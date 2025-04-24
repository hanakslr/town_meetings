import json
from typing import Any, Dict
from pathlib import Path

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
                "required": ["committee_name", "strategy_name", "schema", "values", "code"]
            },
        }
