import asyncio
from typing import Any, Dict

import aiohttp
from anthropic.types import ToolParam
from bs4 import BeautifulSoup

from tools import Tool


class Bs4SiteScraperTool(Tool):
    """A tool class for analyzing webpages using BeautifulSoup."""

    name = "scrape_webpage"

    previous_text_blobs: list[str]
    previous_urls: list[dict[str, str]]
    """Store previous text_blobs"""

    def __init__(self):
        self.previous_text_blobs = []
        self.previous_urls = []

    @classmethod
    def get_tool_definition(cls) -> ToolParam:
        """Return the tool definition that can be passed to Claude."""
        return {
            "name": Bs4SiteScraperTool.name,
            "description": """Scrape a webpage using BeautifulSoup to extract specific elements. Identical information that is in a previous call use may be filtered out and not returned again.
                """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to analyze"},
                    "extract_links": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Filtering term for the link or its display text",
                        },
                        "description": "Array of strings to extract links for  - only links containing these strings as their display text will be included.",
                    },
                    "extract_body_text": {
                        "type": "boolean",
                        "description": "Whether to extract body-like text from the page. This ignores link-like text or nav-like text.",
                    },
                    # "extract_navigation": {
                    #     "type": "boolean",
                    #     "description": "Whether to extract navigation elements",
                    # },
                },
                "required": ["url"],
            },
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given parameters."""
        url = params.get("url")
        extract_links = params.get("extract_links", [])
        extract_body_text = params.get("extract_body_text", False)
        # extract_navigation = params.get("extract_navigation", False)

        # Create SSL context with default settings
        # import ssl
        # ssl_context = ssl.create_default_context()

        # Set custom headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Create session with custom timeout and SSL settings
        timeout = aiohttp.ClientTimeout(total=30)
        # connector = aiohttp.TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            try:
                async with session.get(url, ssl=False) as response:
                    if response.status != 200:
                        return {
                            "error": f"Failed to access URL: HTTP {response.status}"
                        }
                    response_text = await response.text()
            except aiohttp.ClientConnectorError as e:
                print(f"Connection error: {str(e)}")
                return {"error": f"Connection error: {str(e)}"}
            except aiohttp.ClientSSLError as e:
                print(f"SSL error: {str(e)}")
                return {"error": f"SSL error: {str(e)}"}
            except asyncio.TimeoutError:
                print("Request timed out")
                return {"error": "Request timed out"}
            except Exception as e:
                print(f"Request error: {str(e)}")
                return {"error": f"Request error: {str(e)}"}

            soup = BeautifulSoup(response_text, "html.parser")
            result = {"url": url}

            # Extract page title
            result["title"] = soup.title.string if soup.title else "No title"

            # Extract all links if requested
            if extract_links:
                links = []
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    text = a.get_text(strip=True)

                    # Skip links without href or text
                    if not href or not text:
                        continue

                    if not any(
                        [
                            keyword
                            for keyword in extract_links
                            if keyword.lower() in text.lower()
                        ]
                    ):
                        continue

                    # TODO this could be more efficient
                    if any(
                        [
                            prev
                            for prev in self.previous_urls
                            if prev.get("url") == href and prev.get("text") == text
                        ]
                    ):
                        print("Skipping previously found URL")
                    else:
                        self.previous_urls.append({"url": href, "text": text})
                        links.append({"url": href, "text": text})

                result["links"] = links

            # Extract main text if requested
            if extract_body_text:
                tags = ["main", "article", "section", "div", "p"]
                main_elements = soup.find_all(tags)
                main_text = []

                # Skip any elements masquerading as nav-like things
                main_elements = [
                    element
                    for element in main_elements
                    if not any(
                        c in str(element.get("class", []))
                        for c in [
                            "nav",
                            "menu",
                            "footer",
                            "header",
                            "navbar",
                            "footernav",
                        ]
                    )
                ]

                for element in main_elements:
                    text = element.get_text(separator="\n", strip=True)

                    # this isn't the most nested element of the tags we are looking for
                    if element.name != "p" and any(
                        child.name in tags for child in element.find_all(tags)
                    ):
                        continue

                    children = element.find_all()

                    # Skip elements that are entirely composed of links and scripts
                    if len(children) and all(
                        [child.name in ["a", "script"] for child in children]
                    ): 
                        print(text.strip() for text in element.stripped_strings if text not in element.find_all(["a", "script"], recursive=False))
                        print(f"Skipping element that only contains links and scripts: {text}")
                        continue

                    # Skip divs that are inside an <a>
                    if element.name == "div" and element.find_parent("a") is not None:
                        continue

                    if len(text) > 70 and not text in self.previous_text_blobs:
                        main_text.append(text)
                        self.previous_text_blobs.append(text)
                    elif text in self.previous_text_blobs:
                        print(f"Skipping including {len(text)} prev included chars")

                result["main_text"] = main_text

            # Handle navigation elements specifically - this isn't helpful atm
            # if extract_navigation:
            #     nav_elements = soup.select("nav, .nav, .menu, header, .navigation, .navbar")
            #     if nav_elements:
            #         result["navigation"] = []
            #         for nav in nav_elements[:3]:
            #             nav_links = []
            #             for a in nav.find_all("a", href=True):
            #                 nav_links.append(
            #                     {"url": a["href"], "text": a.get_text(strip=True)}
            #                 )
            #             result["navigation"].append({"links": nav_links})

            return result
