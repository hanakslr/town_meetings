import json
import re

from anthropic import AsyncAnthropic, NotGiven
from anthropic.types import Message, ThinkingConfigParam

from tools import Tool


class TaskHandler:
    name: str
    client: AsyncAnthropic
    system_prompt: str
    tools: dict[str, Tool]
    messages: list[Message]

    max_tokens: int

    def __init__(
        self,
        *,
        name: str,
        client: AsyncAnthropic,
        tools: list[type[Tool]],
        system_prompt: str,
    ):
        self.name = name
        self.client = client
        self.messages = []
        self.system_prompt = system_prompt
        self.tools = {t.name: t() for t in tools}

    async def run(
        self,
        *,
        task_prompt: str,
        max_tokens: int,
        thinking: ThinkingConfigParam | NotGiven = NotGiven,
    ):
        self.max_tokens = max_tokens

        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "cache_control": {"type": "ephemeral"},
                        "text": task_prompt,
                    }
                ],
            }
        ]
        response = await self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=self.max_tokens,
            temperature=0,
            thinking=thinking,
            system=self.system_prompt,
            messages=self.messages,
            tools=[tool.get_tool_definition() for tool in self.tools.values()],
            tool_choice={"type": "auto"},
        )

        result = await self.handle_tool_calls(response)

        return result

    async def handle_tool_calls(self, message: Message):
        """Handle any tool calls in a Claude message."""
        content = message.content
        tool_call_found = False

        # Check if there are tool calls to handle
        for item in content:
            if item.type == "tool_use":
                tool_call_found = True

                tool_name = item.name
                tool_params = item.input

                print(f"Running {tool_name}")

                if tool_name in self.tools:
                    tool = self.tools[tool_name]

                    # This is our last stop for structured output.
                    if tool.is_structured_output():
                        return tool_params

                    result = await tool.execute(tool_params)

                    self.messages.append({"role": "assistant", "content": content})
                    self.messages.append(
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
                    print("Response:", json.dumps(result))

                    new_message = await self.client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=self.max_tokens,
                        temperature=0,
                        system=self.system_prompt,
                        messages=self.messages,
                        tools=[
                            tool.get_tool_definition() for tool in self.tools.values()
                        ],
                        tool_choice={"type": "auto"},
                    )

                    print(f"Calling again with {new_message}")

                    # Recursively handle any further tool calls
                    return await self.handle_tool_calls(new_message)

        # If no tool calls or we've completed the process, return the final results
        if not tool_call_found:
            final_content = " ".join(
                [item.text for item in content if item.type == "text"]
            )

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
