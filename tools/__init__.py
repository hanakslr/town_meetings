import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from anthropic import AsyncAnthropic
from anthropic.types import Message, ToolParam


class Tool(ABC):
    name = "abc"  # gotta implement yourself

    @classmethod
    def is_structured_output(cls) -> bool:
        return False

    @classmethod
    @abstractmethod
    def get_tool_definition(cls) -> ToolParam:
        raise Exception("Subclass must implement.")

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise Exception("Subclass must implement")


async def handle_tool_calls(
    client: AsyncAnthropic,
    tools: dict[str, Tool],
    message: Message,
    previous_messages=Optional[list[Message]],
):
    """Handle any tool calls in a Claude message."""
    if previous_messages is None:
        previous_messages = []

    content = message.content
    tool_call_found = False

    # Check if there are tool calls to handle
    for item in content:
        if item.type == "tool_use":
            tool_call_found = True

            tool_name = item.name
            tool_params = item.input

            print(f"Running {tool_name}")

            if tool_name in tools:
                tool = tools[tool_name]

                # This is our last stop for structured output.
                if tool.is_structured_output():
                    return tool_params

                result = await tool.execute(tool_params)

                new_messages = previous_messages.copy()

                new_messages.append({"role": "assistant", "content": content})
                new_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": item.id,
                                "content": json.dumps(result),
                            }
                        ],
                    }
                )

                print("Assistant:", content)
                print("Response:", new_messages[-1])

                new_message = await client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=4000,
                    temperature=0,
                    system="You are an expert in analyzing municipal government websites. You locate information to help keep citizens informed and engaged.",
                    messages=new_messages,
                    tools=[tool.get_tool_definition() for tool in tools.values()],
                    tool_choice={"type": "auto"},
                )

                print(f"Calling again with {new_message}")

                # Recursively handle any further tool calls
                return await handle_tool_calls(client, tools, new_message, new_messages)

    # If no tool calls or we've completed the process, return the final results
    if not tool_call_found:
        final_content = " ".join([item.text for item in content if item.type == "text"])

        # Try to extract structured data from Claude's response
        try:
            # Look for JSON structure in the response
            json_match = re.search(r"\{.*\}", final_content, re.DOTALL)
            if json_match:
                structured_data = json.loads(json_match.group(0))
                return structured_data
            else:
                return {"summary": final_content}
        except Exception as e:
            return {"summary": final_content, "error": str(e)}
