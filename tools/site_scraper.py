from typing import Any, Dict

import requests
import asyncio
import aiohttp
from anthropic.types import ToolParam
from bs4 import BeautifulSoup

from tools import Tool


class Bs4SiteScraperTool(Tool):
    """A tool class for analyzing webpages using BeautifulSoup."""

    def __init__(self):
        pass

    def get_tool_definition(self) -> ToolParam:
        """Return the tool definition that can be passed to Claude."""
        return {
            "name": "scrape_webpage",
            "description": "Scrape a webpage using BeautifulSoup to extract specific elements",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to analyze"},
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to extract specific elements (optional)",
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
                        if href and text:
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
                        if len(text) > 100:
                            main_text.append(text)

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
