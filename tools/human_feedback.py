import json
from typing import Any, Dict

from anthropic.types import ToolParam
from prompt_toolkit import PromptSession

from tools import Tool


class GetHumanFeedbackTool(Tool):
    """A tool class for getting human feedback on structured data."""

    name = "get_human_feedback"

    @classmethod
    def get_tool_definition(cls) -> ToolParam:
        """Return the tool definition that can be passed to Claude."""
        return {
            "name": GetHumanFeedbackTool.name,
            "description": """Get human feedback on structured JSON data by displaying it nicely and prompting for input.
                """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "The structured JSON data to get feedback on",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional prompt to display to the user",
                    },
                },
                "required": ["data"],
            },
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given parameters."""
        data = params.get("data")
        prompt = params.get("prompt", "Please provide any feedback on the above data:")

        # Create a prompt session
        session = PromptSession()

        # Print the data nicely formatted
        print("\nData for review:")

        print(json.dumps(data, indent=2))

        print("\n" + prompt)

        # Get user feedback
        feedback = await session.prompt_async("> ")

        return {"feedback": feedback.strip()}
