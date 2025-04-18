import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from anthropic import Anthropic
from anthropic.types import Message, ToolUseBlock
from dotenv import load_dotenv

from tools import Tool
from tools.site_scraper import Bs4SiteScraperTool

load_dotenv()

SCHEMA_VERSION = 1


@dataclass
class Committee():
    name: str
    url: Optional[str]
    details: Optional[dict[str, Any]]  # TODO type this


class TownWebsiteAnalyzer():
    """Main class for analyzing town websites using Claude and tools."""

    client: Anthropic
    tool_usage: dict[str, ToolUseBlock]
    tools: dict[str, Tool]

    town_name: str
    state: str

    website_url: Optional[str]
    committees: Optional[list[Committee]]

    def __init__(self, town_name: str, state: str):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.tool_usage: dict[str, ToolUseBlock] = {}
        self.tools = {"scrape_webpage": Bs4SiteScraperTool()}
        self.town_name = town_name
        self.state = state
        self.website_url = None
        self.committees = None

    @property
    def __dict__(self) -> dict:
        return {
            "town_name": self.town_name,
            "state": self.state,
            "website_url": self.website_url,
            "committees": self.committees
        }

    def handle_tool_calls(
        self, message: Message, previous_messages=Optional[list[Message]]
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

                self.tool_usage[item.id] = item
                print(f"Running {tool_name} with {tool_params}")

                if tool_name in self.tools:
                    # Execute the tool
                    tool = self.tools[tool_name]
                    result = tool.execute(tool_params)

                    print(f"Got result for {item.id}")

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

                    new_message = self.client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=4000,
                        temperature=0,
                        system="You are an expert in analyzing municipal government websites. You locate information to help keep citizens informed and engaged.",
                        messages=new_messages,
                        tools=[
                            tool.get_tool_definition() for tool in self.tools.values()
                        ],
                        tool_choice={"type": "auto"},
                    )

                    print(f"Calling again with {new_message}")

                    # Recursively handle any further tool calls
                    return self.handle_tool_calls(new_message, new_messages)

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

    def find_town_website(self):
        """Use Claude to find the official website for a town."""
        location = f"{self.town_name}, {self.state}"

        prompt = f"""
        What is the official government website for {location}?
        Please return only the URL without any additional text or explanation.
        """

        message = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=100,
            temperature=0,
            system="You are a helpful research assistant. Answer ONLY with the requested information.",
            messages=[{"role": "user", "content": prompt}],
        )

        website_url = message.content[0].text.strip()

        if website_url:
            print(f"Found website: {website_url}")
            self.website_url = website_url
        else:
            raise Exception("Did not find website URL")

    def find_town_orgs(self):
        try:
            # Initial message to Claude with tools
            initial_messages = [
                {
                    "role": "user",
                    "content": f"""
                The official town website for {self.town_name}, {self.state} is {self.website_url}
                Analyze the town website to find:
                
                1. All boards, committees, and commissions
                2. The URL of a webpage with information about that group.
                
                Use the scrape_webpage tool to help with this analysis. Start by examining the main page,
                then look for navigation elements that might lead to committees or government sections.

                Each organization may store their information completely differently. 
                
                Return your findings as a structured JSON with this format:
                {{
                  "committees": [
                    {{
                      "name": "Committee Name",
                      "url": "URL to committee page"
                    }}
                  ]
                }}
                """,
                }
            ]

            # Create message with tool that can use BeautifulSoup
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                temperature=0,
                system="You are an expert in analyzing municipal government websites. Use the provided tools to extract information about town committees. You have access to tools, but only use them when necessary. If a tool is not required, respond as normal.",
                messages=initial_messages,
                tools=[tool.get_tool_definition() for tool in self.tools.values()],
                tool_choice={"type": "auto"},
            )

            # Process the message and handle tool calls
            result = self.handle_tool_calls(response, initial_messages)

            committees = result.get("committees", None)

            if committees:
                self.committees = committees
            else:
                raise Exception("Could not find committees")

        except Exception as e:
            return {"error": str(e)}

    def find_org_details(self, comittee: Committee):
        """Given a committee, commission, or board name and its website URL
        extract details about when it meets and its agendas"""

        try:
            # Initial message to Claude with tools
            initial_messages = [
                {
                    "role": "user",
                    "content": f"""

                The official website for the {comittee.name} of {self.town_name}, {self.state} is {comittee.url}. This is a municipal board, committee, or commission.

                Analyze the website to find the meeting schedule and location for this group, as well as how and where the agendas are
                stored. Public municipal bodies are required by law to publish their agendas and your job is to report where they can be found.

                Some groups meet regularly and others only meet as needed. If the information is not readily available, just leave what cannot 
                be found empty. If they do meet regularly the information will be easily found.

                Each group handles this differently.
                
                Return your findings as a structured JSON with this format:
                {{
                    "schedule": "When the group regularly meets - as a string, like 1st and 3rd Tuesdays at 7pm",
                    "meeting_location": "Where the group typically meets - as a string".
                    "agendas": {{
                        "type": "pdf-links|embedded-html|document-library|calendar|unknown",
                        "location": "URL where documents are found",
                        "pattern": "Pattern for identifying documents (if applicable)",
                        "notes": "Additional information"
                      }}
                }}
                """,
                }
            ]

            # Create message with tool that can use BeautifulSoup
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                temperature=0,
                system="You are an expert in analyzing municipal government websites to help citizens stay informed and engaged on whats happens when. Use the provided tools to extract information about town committees. You have access to tools, but only use them when necessary. If a tool is not required, respond as normal.",
                messages=initial_messages,
                tools=[tool.get_tool_definition() for tool in self.tools.values()],
                tool_choice={"type": "auto"},
            )

            # Process the message and handle tool calls
            result = self.handle_tool_calls(response, initial_messages)

            return result

        except Exception as e:
            return {"error": str(e)}

    def run_workflow(self) -> Dict[str, Any]:
        """Run the full town website analysis workflow."""
        if not self.website_url:
            self.find_town_website()

        if not self.committees:
            self.find_town_orgs()

        # for committee in self.committees:
        #     if not committee.details:
        #         self.find_org_details(committee)


if __name__ == "__main__":
    import os

    town_name = "Williston"
    state = "VT"
    resume_latest = True

    # Make a {state}_{town_name} dir if it doesn't already exist
    directory_path = f"output/{state}/{town_name}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    analyzer = TownWebsiteAnalyzer(town_name=town_name, state=state)

    # Pick up where we left off
    if resume_latest:
        import glob

        files = glob.glob(f"{directory_path}/s{SCHEMA_VERSION}_*.json")

        if files:
            latest = max(files, key=os.path.getctime)
            print(f"Resuming from: {latest}")
            previous_result = json.load(latest)
            print(previous_result)
            raise Exception("All done")

    try:
        analyzer.run_workflow()
    finally:
        # Save results to file
        with open(
            f"{directory_path}/s{SCHEMA_VERSION}_{datetime.now().isoformat()}.json", "w"
        ) as f:
            json.dump(analyzer.__dict__, f, indent=2, default=lambda o: o.__dict__)

        print("Analysis complete.")
