import asyncio
from typing import Any, Dict

import aiohttp
from anthropic.types import ToolParam
from bs4 import BeautifulSoup

from tools import Tool


class Bs4SiteScraperTool(Tool):
    """A tool class for analyzing webpages using BeautifulSoup."""


    previous_text_blobs: list[str]
    """Store previous text_blobs"""

    def __init__(self):
        self.previous_text_blobs = []

    @classmethod
    def get_tool_definition(cls) -> ToolParam:
        """Return the tool definition that can be passed to Claude."""
        return {
            "name": "scrape_webpage",
            "description": """Scrape a webpage using BeautifulSoup to extract specific elements. This tool returns
                a lot of information so define the extraction filters whenever possible.
                """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to analyze"},
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to extract specific elements (optional)",
                    },
                    "extraction_filters": {
                        "type": "object",
                        "properties": {
                            "links": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "Filtering term for links"
                                },
                                "description": "Array of strings to filter links by (only links containing these strings as their display text will be included)",
                            },
                        }
                    },
                    "extract_links": {
                        "type": "boolean",
                        "description": "Whether to extract all links from the page. Extracting links is useful in determining what page things are on.",
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "Whether to extract all text from the page. This is useful for gaining specific information once it has been located, but is expensive if you don't really need it because it returns so much content.",
                    },
                    "extract_navigation": {
                        "type": "boolean",
                        "description": "Whether to extract navigation elements",
                    },
                },
                "required": ["url"],
            },
        }
        

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given parameters."""
        url = params.get("url")
        selector = params.get("selector")
        extract_links = params.get("extract_links", False)
        extract_text = params.get("extract_text", False)
        extract_navigation = params.get("extract_navigation", False)
        extraction_filters = params.get("extraction_filters", {})

        if extract_links and not extraction_filters:
            raise Exception("Required extraction filters")

        try:
            # Create SSL context with default settings
            import ssl
            ssl_context = ssl.create_default_context()
            
            # Set custom headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Create session with custom timeout and SSL settings
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                try:
                    async with session.get(url, ssl=False) as response:
                        if response.status != 200:
                            return {"error": f"Failed to access URL: HTTP {response.status}"}
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

                # Extract by selector if provided
                if selector:
                    elements = soup.select(selector)
                    result["selector_results"] = []
                    for element in elements:
                        result["selector_results"].append(
                            {"text": element.get_text(strip=True), "html": str(element)}
                        )

                # Extract all links if requested
                if extract_links:
                    links = []
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        text = a.get_text(strip=True)
                        # Skip links without href or text
                        if not href or not text:
                            continue
                            
                        # Apply filters if provided
                        if extraction_filters.get("links", None):
                            # Check if any filter string is in either the href or text
                            if not any(filter_str.lower() in href.lower() or 
                                      filter_str.lower() in text.lower() 
                                      for filter_str in extraction_filters.get("links")):
                                continue
                                
                        links.append({"url": href, "text": text})
                    result["links"] = links

                # Extract main text if requested
                if extract_text:
                    main_elements = soup.find_all(["main", "article", "section", "div"])
                    main_text = []

                    for element in main_elements:
                        if any(
                            c in str(element.get("class", []))
                            for c in ["nav", "menu", "footer", "header"]
                        ):
                            continue

                        text = element.get_text(separator="\n", strip=True)
                        if len(text) > 100 and not text in self.previous_text_blobs:
                            main_text.append(text)
                            self.previous_text_blobs.append(text)
                        elif text in self.previous_text_blobs:
                            print(f"Skipping including {len(text)} prev included chars")

                    result["main_text"] = main_text[:5]

                # Handle navigation elements specifically
                if extract_navigation:
                    nav_elements = soup.select("nav, .nav, .menu, header, .navigation")
                    if nav_elements:
                        result["navigation"] = []
                        for nav in nav_elements[:3]:
                            nav_links = []
                            for a in nav.find_all("a", href=True):
                                nav_links.append(
                                    {"url": a["href"], "text": a.get_text(strip=True)}
                                )
                            result["navigation"].append({"links": nav_links})

                return result

        except Exception as e:
            print(str(e))
            return {"error": f"Error analyzing webpage: {str(e)}"}
