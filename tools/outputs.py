from typing import Any, Dict, override

from anthropic.types import ToolParam

from tools import Tool


class StructuredOutputTool(Tool):
    """A tool that only defines an input schema for the purpose of getting the LLM to format its output in a certain structure."""

    @classmethod
    def is_structured_output(cls) -> bool:
        return True
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise Exception("This should not be called")


class AllOrgsOutputTool(StructuredOutputTool):
    name = "all_orgs_summary"
    
    @classmethod
    def get_tool_definition(cls) -> ToolParam:
        return {
            "name": AllOrgsOutputTool.name,
            "description": "Record a summary of a municipality - a URL where to find agendas & minutes (if exists), and a list of each of the boards/committees/commissions and their respective overview pages",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agendas_url": {
                        "type": "string",
                        "description": "URL to the highest level page that has Agendas and/or Minutes for the municipality. Leave empty if not found."
                    },
                    "committees": {
                        "type": "array",
                        "description": "A list of all of the committees/commission/boards the municipality has",
                        "items": {
                             "type": "object",
                             "properties": {
                                  "name": {
                                       "type": "string",
                                       "description": "The committee name"
                                  },
                                  "overview_url": {
                                       "type": "string",
                                       "description": "URL to the committee overview page. This general contains information about the committee and is not specific to the agendas - it may contain the members, purpose of the group, or when they meet."
                                  },
                                  "agendas_url": {
                                      "type": "string",
                                      "description": "URL of the highest level page that has Agendas/Minutes for this specific committee. If there is no page specifically for this committee (instead only a joint one for the entire municipalities minutes), leave blank."
                                  }
                             }
                        }

                    }
                },
            }
        }

class CommitteeDetailsOutputTool(StructuredOutputTool): 
    name = "committee_meeting_times_summary"
    @classmethod
    def get_tool_definition(cls) -> ToolParam:
        return {
            "name": CommitteeDetailsOutputTool.name,
            "description": "Record a summary of when a municipal group or committee meets and where to find agenda information using well structured JSON.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "schedule": {
                        "type": "string",
                        "description": "A concise user friendly description of when the group meets. Like '1st and 3rd Tuesdays at 7pm' or 'As needed'"
                    },
                    "schedule_cron": {
                        "type": "string",
                        "description": "A cron representation of the meeting schedule if the group meets regularly. If not regular meetings, this should be left null"
                    },
                    "meeting_location": {
                        "type": "string",
                        "description": "Where the group regularly meets. If no regular meetings, leave null. If this information is not readily available, leave null."
                    },
                    "agendas": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL where the agenda documents can be found."
                            },
                            "format": {
                                "type": "string",
                                "enum": ["document-links","embedded-html","unknown","other"],
                                "description": " How the agenda is stored. document-links: links to external files. embedded-html: links to webpage that displays the agenda."
                            },
                            "notes": {
                                "type": "string",
                                "description": "Concise additional information for locating the agendas."
                            }

                        }
                    }
                }
            }
        }
