from typing import Any, Dict, override
from anthropic.types import ToolParam

from tools import Tool


class CommitteeDetailsOutputTool(Tool):    
    @classmethod
    def is_structured_output(cls) -> bool:
        return True

    @classmethod
    def get_tool_definition(cls) -> ToolParam:
        return {
            "name": "committee_meeting_times_summary",
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

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise Exception("This should not be called")
