import json
import re
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

from anthropic import NOT_GIVEN, AsyncAnthropic, NotGiven
from anthropic.types import Message, ThinkingConfigParam

from tools import Tool


class TaskHandler:
    name: str
    client: AsyncAnthropic
    system_prompt: str
    tools: dict[str, Tool]
    messages: list[Message]
    thinking: ThinkingConfigParam | NotGiven

    max_tokens: int
    _original_sigint_handler: Optional[callable]

    def __init__(
        self,
        *,
        name: str,
        client: AsyncAnthropic,
        tools: list[type[Tool]],
        system_prompt: str,
        thinking: ThinkingConfigParam | NotGiven = NOT_GIVEN,
    ):
        self.name = name
        self.client = client
        self.messages = []
        self.system_prompt = system_prompt
        self.tools = {t.name: t() for t in tools}
        self.thinking = thinking
        self.max_tokens = 0
        self._original_sigint_handler = None

    def _save_messages(self):
        """Save current messages to a file."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("output/interrupted_tasks")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().isoformat()
            filename = output_dir / f"{self.name}_{timestamp}.json"

            # Save messages to file
            with open(filename, "w") as f:
                json.dump(
                    {
                        "name": self.name,
                        "system_prompt": self.system_prompt,
                        "messages": self.messages,
                        "interrupted_at": timestamp,
                    },
                    f,
                    indent=2,
                )

            print(f"\nTask interrupted. Messages saved to {filename}")
        except Exception as e:
            print(f"Error saving interrupted task: {e}")

    def _sigint_handler(self, signum, frame):
        """Handle SIGINT (Ctrl+C) by saving messages and restoring original handler."""
        self._save_messages()
        if self._original_sigint_handler:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            self._original_sigint_handler(signum, frame)

    async def run(self, *, task_prompt: str, max_tokens: int):
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

        # Set up signal handler
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)

        try:
            response = await self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=self.max_tokens,
                temperature=0 if self.thinking == NOT_GIVEN else 1,
                thinking=self.thinking,
                system=self.system_prompt,
                messages=self.messages,
                tools=[tool.get_tool_definition() for tool in self.tools.values()],
                tool_choice={"type": "auto"},
            )

            result = await self.handle_tool_calls(response)
            return result
        finally:
            # Restore original signal handler
            if self._original_sigint_handler:
                signal.signal(signal.SIGINT, self._original_sigint_handler)

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

                print(f"\n\nRunning {tool_name}\n")

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
                    print("\nResponse:", json.dumps(result, indent=2))

                    new_message = await self.client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=self.max_tokens,
                        temperature=0 if self.thinking == NOT_GIVEN else 1,
                        thinking=self.thinking,
                        system=self.system_prompt,
                        messages=self.messages,
                        tools=[
                            tool.get_tool_definition() for tool in self.tools.values()
                        ],
                        tool_choice={"type": "auto"},
                    )

                    print(f"\nCalling again with {new_message.to_json(indent=2)}")

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
