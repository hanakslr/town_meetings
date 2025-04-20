import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from anthropic import AsyncAnthropic
from anthropic.types import ToolUseBlock
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.validation import ValidationError, Validator

from tools import Tool, handle_tool_calls
from tools.human_feedback import GetHumanFeedbackTool
from tools.outputs import AllOrgsOutputTool, OrgMeetingDetailsOutputTool
from tools.site_scraper import Bs4SiteScraperTool

load_dotenv()

SCHEMA_VERSION = 2

GENERAL_TOOLS = {Bs4SiteScraperTool.name: Bs4SiteScraperTool()}

# Predefined skip reasons
SKIP_REASONS = [
    "Overview URL is minutes URL",
    "No meeting information available",
    "Inactive",
    "Not important",
    "Information is outdated",
    "Other (specify)",
]


class SkipReasonValidator(Validator):
    def validate(self, document):
        text = document.text.strip()
        if not text:
            raise ValidationError(message="Please enter a reason or 'c' to continue")
        if text.lower() == "c":
            return
        try:
            index = int(text)
            if index < 1 or index > len(SKIP_REASONS):
                raise ValidationError(
                    message=f"Please enter a number between 1 and {len(SKIP_REASONS)} or 'c' to continue"
                )
        except ValueError:
            raise ValidationError(
                message="Please enter a valid number or 'c' to continue"
            )


@dataclass
class Committee:
    name: str
    overview_url: Optional[str]
    agendas_url: Optional[str]
    skip_reason: Optional[str]
    meeting_details: Optional[dict[str, Any]] = None
    fetching_strategy: Optional[dict[str, Any]] = None
    details: Optional[dict[str, Any]] = None


class TownWebsiteAnalyzer:
    """Main class for analyzing town websites using Claude and tools."""

    client: AsyncAnthropic
    tool_usage: dict[str, ToolUseBlock]

    town_name: str
    state: str

    website_url: Optional[str]
    agendas_url: Optional[str]
    committees: Optional[list[Committee]]

    def __init__(self, town_name: str, state: str):
        self.client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.tool_usage: dict[str, ToolUseBlock] = {}
        self.town_name = town_name
        self.state = state
        self.website_url = None
        self.agendas_url = None
        self.committees = None

    @property
    def __dict__(self) -> dict:
        return {
            "town_name": self.town_name,
            "state": self.state,
            "website_url": self.website_url,
            "agendas_url": self.agendas_url,
            "committees": self.committees,
        }

    def resume_from(self, previous_result):
        for key, value in previous_result.items():
            if key == "committees" and value is not None:
                self.committees = []
                for c in value:
                    try:
                        committee_fields = {
                            field: None for field in Committee.__dataclass_fields__
                        }
                        committee_data = {
                            field: c.get(field) for field in committee_fields
                        }
                        self.committees.append(Committee(**committee_data))
                    except Exception as e:
                        print(f"Warning: Error processing committee data: {e}")
                        continue
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

            tools = {**GENERAL_TOOLS, AllOrgsOutputTool.name: AllOrgsOutputTool()}

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
            result = await handle_tool_calls(
                self.client, tools, response, initial_messages
            )

            self.committees = result.get("committees", None)
            self.agendas_url = result.get("agendas_url", None)

        except Exception as e:
            return {"error": str(e)}

    async def find_org_meeting_details(self, committee: Committee):
        initial_messages = [
            {
                "role": "user",
                "content": f"""
                    There is a municipal group, the {committee.name} for {self.town_name}, {self.state}. This is a municipal board, committee, or commission.

                    There is an overview page that gives details for the organization at {committee.overview_url}.
        
                    Find the meeting schedule and location for this group.

                    Some groups meet regularly and others only meet as needed. If the schedule and location information is not readily available, just leave what cannot 
                    be found empty. If they do meet regularly the information will be easily found. No need to check specific documents.

                    Return your findings using the {OrgMeetingDetailsOutputTool.name}
                """,
            }
        ]

        tools = {**GENERAL_TOOLS, OrgMeetingDetailsOutputTool.name: OrgMeetingDetailsOutputTool()}

        # Create message with tool that can use BeautifulSoup
        response = await self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=0,
            system="You are an expert in analyzing municipal government websites. Use the provided tools to extract information about town committees. You have access to tools, but only use them when necessary. If a tool is not required, respond as normal.",
            messages=initial_messages,
            tools=[tool.get_tool_definition() for tool in tools.values()],
            tool_choice={"type": "auto"},
        )

        # Process the message and handle tool calls
        result = await handle_tool_calls(
            self.client, tools, response, initial_messages
        )

        committee.meeting_details = result

    async def find_org_agenda_fetching_strategy(self, comittee: Committee):
        """Given a committee, commission, or board name and its website URL
        generate a fetching strategy for its agendas."""
        print(f"Finding details for {comittee=}")

        prompt = f"""

            There is a municipal group, the {comittee.name} for {self.town_name}, {self.state}. This is a municipal board, committee, or commission.

            {f"They meet: {comittee.meeting_details.get("schedule", None)}" if comittee.meeting_details.get("schedule", None) else ""}
            There is an overview page that gives details for the organization at {comittee.overview_url}.
            {f"There is an page that gives access to their meeting agendas at: {comittee.agendas_url}" if comittee.agendas_url else ""}
            {f"The meeting agendas for all municipal groups can be found at: {self.agendas_url}" if self.agendas_url else ""}.

            Generate a machine-consumable strategy for locating all of the groups agendas.
            Public municipal bodies are required by law to publish their agendas. These will
            only ever be referred to as "agendas" or "minutes" and will be available somewhere on the page, either directly or via a link.

            Each group handles this differently and you should produce a reliable fetching strategy represented in a JSON schema.

            You are allowed to define a custom schema for each committee or board, based on how their data is structured — but your schema must be programmatically useful.

            Keep the schemas verbosity minimal, and the complexity as high as needed to get complete information — only include fields that are required to fetch the data. Do not return markdown, explanation text, or full paragraphs. 
            This output will be passed directly to a downstream code system. Downstream, BeautifulSoup (among other tools) could be used for retrieval.

            The fetching_strategy should be a JSON object like this:

            {{
                "strategy_type": "shared" | "individual" | "not_found", # or something else that makes sense
                "strategy_name": "yearly_archive" | "embedded-html-links" | "filter-table" # these are examples. Make up something that makes sense.
                "schema": {{
                    "field_1": "description of how this field is used",
                    "field_2": "..."
                }},
                "values": {{
                    "field_1": "actual value for this committee",
                    "field_2": "..."
                }},
                "notes": "Optional clarifications or edge cases",
                "code_snippet": "Python code snippet to get the agendas based on the schema presented, as minimal as possible - it does not need to be a working example. It should be specific to the strategy_name and not contain any hard coded references to this specific committee."
            }}

            Propose your strategy for feedback using the {GetHumanFeedbackTool.name} tool, and make as many iterations as needed.

            After recieving the go ahead from human feedback that it looks good, return your fetching strategy as structured JSON as specifief above.
        """

        # Initial message to Claude with tools
        initial_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "cache_control": {"type": "ephemeral"},
                        "text": prompt,
                    }
                ],
            }
        ]

        # Return your findings using the {CommitteeDetailsOutputTool.name} tool.

        system_prompt = """
          You are a web scraping strategist who analyzes municipal websites and proposes machine-consumable strategies for 
          locating agendas and information about boards and committees. You work for a system that will automatically scrape agendas and minutes for each board or committee

          You are allowed to define a custom schema for each committee or board, based on how their data is structured — but your schema must be programmatically useful.

          Use the provided tools to extract information about town committees. You have access to tools, but only use them when necessary. 
          If a tool is not required, respond as normal.
        """

        tools: dict[str, Tool] = {
            **GENERAL_TOOLS,
            GetHumanFeedbackTool.name: GetHumanFeedbackTool(),
            # CommitteeDetailsOutputTool.name: CommitteeDetailsOutputTool()
        }

        # Create message with tool that can use BeautifulSoup
        response = await self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
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

        session = PromptSession()
        for committee in self.committees:
            if (not committee.meeting_details or not committee.fetching_strategy) and not committee.skip_reason:
                # Display committee info and skip reasons
                print(f"\nCommittee: {committee.name}")
                print("URL:", committee.overview_url)
                print("Agenda URL:", committee.agendas_url)
                print("\nSkip reasons:")
                for i, reason in enumerate(SKIP_REASONS, 1):
                    print(f"{i}. {reason}")
                print("c. Continue with analysis")
                print("x. Exit")

                # Get user input
                response = (
                    (
                        await session.prompt_async(
                            "Enter a number to skip or 'c' to continue: ",
                            validator=SkipReasonValidator(),
                        )
                    )
                    .strip()
                    .lower()
                )

                if response == "x":
                    return

                if response != "c":
                    index = int(response) - 1
                    if index == len(SKIP_REASONS) - 1:  # "Other" option
                        custom_reason = (
                            await session.prompt_async("Please specify the reason: ")
                        ).strip()
                        committee.skip_reason = custom_reason
                    else:
                        committee.skip_reason = SKIP_REASONS[index]
                    continue
                
                if not committee.meeting_details:
                    await self.find_org_meeting_details(committee)
                if not committee.fetching_strategy:
                    await self.find_org_agenda_fetching_strategy(committee)


if __name__ == "__main__":
    import argparse
    import os

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze a town website for meeting information."
    )
    parser.add_argument("town_name", help="Name of the town to analyze")
    parser.add_argument("state", help="Two-letter state code (e.g., VT, MA, NY)")
    parser.add_argument(
        "--no-resume", action="store_true", help="Do not resume from latest saved state"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate state code
    if len(args.state) != 2 or not args.state.isalpha():
        parser.error("State must be a valid 2-letter code (e.g., VT, MA, NY)")

    # Convert state to uppercase
    state = args.state.upper()
    town_name = args.town_name
    resume_latest = not args.no_resume

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
                with open(latest, "r") as f:
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
