import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from anthropic import AsyncAnthropic, NotGiven
from anthropic.types import Message, ToolParam, ToolUseBlock, tool_param
from dotenv import load_dotenv

from tools import Tool, handle_tool_calls
from tools.outputs import AllOrgsOutputTool, CommitteeDetailsOutputTool
from tools.site_scraper import Bs4SiteScraperTool

load_dotenv()

SCHEMA_VERSION = 2

GENERAL_TOOLS = {Bs4SiteScraperTool.name: Bs4SiteScraperTool()}


@dataclass
class Committee():
    name: str
    url: Optional[str]
    details: Optional[dict[str, Any]]  = None


class TownWebsiteAnalyzer():
    """Main class for analyzing town websites using Claude and tools."""

    client: AsyncAnthropic
    tool_usage: dict[str, ToolUseBlock]

    town_name: str
    state: str

    website_url: Optional[str]
    committees: Optional[list[Committee]]

    def __init__(self, town_name: str, state: str):
        self.client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.tool_usage: dict[str, ToolUseBlock] = {}
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

    def resume_from(self, previous_result):
        for key, value in previous_result.items():
            if key == "committees" and value is not None:
                self.committees = []
                for c in value:
                    committee_data = {
                        "name": c["name"],
                        "url": c["url"],
                    }
                    if "details" in c:
                        committee_data["details"] = c["details"]
                    self.committees.append(Committee(**committee_data))
            elif hasattr(self, key):
                setattr(self, key, value)


    async def find_town_website(self):
        """Use Claude to find the official website for a town."""
        location = f"{self.town_name}, {self.state}"

        prompt = f"""
        What is the official government website for {location}?
        Please return only the URL without any additional text or explanation.
        """

        message = await self.client.messages.create(
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

    async def find_town_orgs(self):
        try:
            # Initial message to Claude with tools
            initial_messages = [
                {
                    "role": "user",
                    "content": f"""
                The official town website for {self.town_name}, {self.state} is {self.website_url}
                Analyze the town website to find:
                
                1. The URL for agendas and/or minutes for all orgs. This is not specific to one org. It may not exist.
                2. All boards, committees, and commissions
                3. The URL of a webpage with specific information about that group.
                
                Use the {Bs4SiteScraperTool.name} tool to help with this analysis. Start by examining the main page,
                then look for navigation elements or links that might lead to committees or government sections.

                Municipal website may have all of the agendas on a single page, or they may have a single page that links out to each group, or there
                may be a separate agendas page for each group.

                Each organization may have a different page structure. 
                
                Return your findings using the {AllOrgsOutputTool.name}
                """,
                }
            ]

            tools = {
                **GENERAL_TOOLS,
                AllOrgsOutputTool.name: AllOrgsOutputTool()
            }

            # Create message with tool that can use BeautifulSoup
            response = await self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                temperature=0,
                system="You are an expert in analyzing municipal government websites. Use the provided tools to extract information about town committees. You have access to tools, but only use them when necessary. If a tool is not required, respond as normal.",
                messages=initial_messages,
                tools=[tool.get_tool_definition() for tool in tools.values()],
                tool_choice={"type": "auto"},
            )

            # Process the message and handle tool calls
            result = await handle_tool_calls(self.client, tools, response, initial_messages)

            if result:
                self.committees = result
            else:
                raise Exception("Could not find committees")

        except Exception as e:
            return {"error": str(e)}

    async def find_org_details(self, comittee: Committee):
        """Given a committee, commission, or board name and its website URL
        extract details about when it meets and its agendas"""
        print(f"Finding details for {comittee=}")

        # Initial message to Claude with tools
        initial_messages = [
            {
                "role": "user",
                "content": f"""

            The official webpage for the {comittee.name} of {self.town_name}, {self.state} is {comittee.url}. This is a municipal board, committee, or commission.

            Analyze the webpage to find the meeting schedule and location for this group, as well as how and where the agendas are
            stored. Public municipal bodies are required by law to publish their agendas and your job is to report where they can be found. These will
            only ever be referred to as "agendas" or "minutes" and will be available somewhere on the page, either directly or via a link.

            Some groups meet regularly and others only meet as needed. If the schedule and location information is not readily available, just leave what cannot 
            be found empty. If they do meet regularly the information will be easily found. No need to check specific documents.

            Each group handles this differently.
            
            Return your findings using the {CommitteeDetailsOutputTool.name} tool.
            """,
            }
        ]

        tools: dict[str, Tool] = {
            **GENERAL_TOOLS,
            CommitteeDetailsOutputTool.name: CommitteeDetailsOutputTool()
        }

        # Create message with tool that can use BeautifulSoup
        response = await self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0,
            system="You are an expert in analyzing municipal government websites to help citizens stay informed and engaged on whats happens when. Use the provided tools to extract information about town committees. You have access to tools, but only use them when necessary. If a tool is not required, respond as normal.",
            messages=initial_messages,
            tools=[tool.get_tool_definition() for tool in tools.values()],
            tool_choice={"type": "auto"},
        )

        # Process the message and handle tool calls
        result = await handle_tool_calls(self.client, tools, response, initial_messages)

        comittee.details = result


    async def run_workflow(self) -> Dict[str, Any]:
        """Run the full town website analysis workflow."""
        if not self.website_url:
            await self.find_town_website()

        if not self.committees:
            await self.find_town_orgs()

        raise Exception("stop")

        for committee in self.committees:
            if not committee.details:
                await self.find_org_details(committee)
                await asyncio.sleep(30)

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
    try:
        # Pick up where we left off
        if resume_latest:
            import glob

            files = glob.glob(f"{directory_path}/s{SCHEMA_VERSION}_*.json")

            if files:
                latest = max(files, key=os.path.getctime)
                print(f"Resuming from: {latest}")
                with open(latest, 'r') as f:
                    previous_result = json.load(f)
                analyzer.resume_from(previous_result)

        asyncio.run(analyzer.run_workflow())
    finally:
        # Save results to file
        with open(
            f"{directory_path}/s{SCHEMA_VERSION}_{datetime.now().isoformat()}.json", "w"
        ) as f:
            json.dump(analyzer.__dict__, f, indent=2, default=lambda o: o.__dict__)

        print("Analysis complete.")
