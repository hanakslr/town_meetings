import requests
import json
import os
import re

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from anthropic import Anthropic
from typing import Dict, Any, List, Optional

load_dotenv()

class WebpageAnalyzerTool:
    """A tool class for analyzing webpages using BeautifulSoup."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return the tool definition that can be passed to Claude."""
        return {
            "name": "scrape_webpage",
            "description": "Scrape a webpage using BeautifulSoup to extract specific elements",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to analyze"
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to extract specific elements (optional)"
                    },
                    "extract_links": {
                        "type": "boolean",
                        "description": "Whether to extract all links from the page. Extracting links is useful in determining what page things are on."
                    },
                    "extract_text": {
                        "type": "boolean", 
                        "description": "Whether to extract all text from the page. This is useful for gaining specific information once it has been located, but is expensive if you don't really need it because it returns so much content."
                    },
                    "extract_navigation": {
                        "type": "boolean",
                        "description": "Whether to extract navigation elements"
                    }
                },
                "required": ["url"]
            }
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given parameters."""
        url = params.get("url")
        selector = params.get("selector")
        extract_links = params.get("extract_links", False)
        extract_text = params.get("extract_text", False)
        extract_navigation = params.get("extract_navigation", False)
        
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return {"error": f"Failed to access URL: HTTP {response.status_code}"}
                
            soup = BeautifulSoup(response.text, 'html.parser')
            result = {"url": url}
            
            # Extract page title
            result["title"] = soup.title.string if soup.title else "No title"
            
            # Extract by selector if provided
            if selector:
                elements = soup.select(selector)
                result["selector_results"] = []
                for element in elements:
                    result["selector_results"].append({
                        "text": element.get_text(strip=True),
                        "html": str(element)
                    })
            
            # Extract all links if requested
            if extract_links:
                links = []
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    text = a.get_text(strip=True)
                    if href and text:
                        links.append({
                            "url": href,
                            "text": text
                        })
                result["links"] = links
                print("Extracted links")
            
            # Extract main text if requested
            if extract_text:
                main_elements = soup.find_all(['main', 'article', 'section', 'div'])
                main_text = []
                
                for element in main_elements:
                    if any(c in str(element.get('class', [])) for c in ['nav', 'menu', 'footer', 'header']):
                        continue
                        
                    text = element.get_text(separator='\n', strip=True)
                    if len(text) > 100:
                        main_text.append(text)
                
                result["main_text"] = main_text[:5]
                print("Extracted text")
            
            # Handle navigation elements specifically
            if extract_navigation:
                nav_elements = soup.select('nav, .nav, .menu, header, .navigation')
                if nav_elements:
                    result["navigation"] = []
                    for nav in nav_elements[:3]:
                        nav_links = []
                        for a in nav.find_all('a', href=True):
                            nav_links.append({
                                "url": a['href'],
                                "text": a.get_text(strip=True)
                            })
                        result["navigation"].append({
                            "links": nav_links
                        })
                print("Extracted nav")
            
            return result
            
        except Exception as e:
            print(str(e))
            return {"error": f"Error analyzing webpage: {str(e)}"}


class TownWebsiteAnalyzer:
    """Main class for analyzing town websites using Claude and tools."""
    
    def __init__(self):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.tools = {
            "scrape_webpage": WebpageAnalyzerTool()
        }
    
    def handle_tool_calls(self, message, previous_messages=None):
        """Handle any tool calls in a Claude message."""
        if previous_messages is None:
            previous_messages = []
            
        content = message.content
        tool_call_found = False
        
        # Check if there are tool calls to handle
        for item in content:
            if item.type == "tool_use":
                print(f"Found tool use: {item}")
                tool_call_found = True
                
                tool_name = item.name
                tool_params = item.input

                print(f"Running {tool_name} with {tool_params}")
                
                if tool_name in self.tools:
                    # Execute the tool
                    tool = self.tools[tool_name]
                    result = tool.execute(tool_params)

                    print(f"Got result for {item.id}")

                    new_messages = previous_messages.copy()
                    new_messages.append({"role": "assistant", "content": content})
                    new_messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": item.id, "content": json.dumps(result)}]})
                    
                    # Send the tool result back to Claude
                    new_message = self.client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=4000,
                        temperature=0,
                        system="You are an expert in analyzing municipal government websites. You locate information to help keep citizens informed and engaged.",
                        messages=new_messages,
                        tools=[tool.get_tool_definition() for tool in self.tools.values()],
                        tool_choice={"type": "auto"}
                    )

                    print(f"Calling again with {new_message}")
                    
                    # Recursively handle any further tool calls
                    return self.handle_tool_calls(new_message, new_messages)
        
        # If no tool calls or we've completed the process, return the final results
        if not tool_call_found:
            final_content = " ".join([item.text for item in content if item.type == "text"])
            
            # Try to extract structured data from Claude's response
            try:
                # Look for JSON structure in the response
                json_match = re.search(r'\{.*\}', final_content, re.DOTALL)
                if json_match:
                    structured_data = json.loads(json_match.group(0))
                    return structured_data
                else:
                    return {"summary": final_content}
            except Exception as e:
                return {"summary": final_content, "error": str(e)}

    def find_town_website(self, town_name, state=None):
        """Use Claude to find the official website for a town."""
        location = f"{town_name}, {state}" if state else town_name
        
        prompt = f"""
        What is the official government website for {location}?
        Please return only the URL without any additional text or explanation.
        """
        
        message = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=100,
            temperature=0,
            system="You are a helpful research assistant. Answer ONLY with the requested information.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        website_url = message.content[0].text.strip()
        print(f"Found website: {website_url}")
        return website_url
    
    def run_workflow(self, town_name: str, state: Optional[str] = None) -> Dict[str, Any]:
        """Run the full town website analysis workflow."""
        website_url = self.find_town_website(town_name, state)
        try:
            # Initial message to Claude with tools
            initial_messages = [
                {"role": "user", "content": f"""
                The official town website for {town_name}, {state} is {website_url}
                Analyze the town website to find:
                
                1. All boards, committees, and commissions
                2. The URL of a webpage with information about that group.
                3. When their meetings happen. Such as "1st and 3rd Tuesdays at 7pm"
                
                Use the scrape_webpage tool to help with this analysis. Start by examining the main page,
                then look for navigation elements that might lead to committees or government sections.

                Each organization may store their information completely differently. 
                
                Return your findings as a structured JSON with this format:
                {{
                  "town": "{town_name}",
                  "website": "website url",
                  "committees": [
                    {{
                      "name": "Committee Name",
                      "url": "URL to committee page",
                      "schedule": "Description of when the committee meets"
                    }}
                  ]
                }}
                """}
            ]
            

                    #               "agendas": {{
                    #     "type": "pdf-links|embedded-html|document-library|calendar|unknown",
                    #     "location": "URL where documents are found",
                    #     "pattern": "Pattern for identifying documents (if applicable)",
                    #     "notes": "Additional information"
                    #   }}
            # Create message with tool that can use BeautifulSoup
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                temperature=0,
                system="You are an expert in analyzing municipal government websites. Use the provided tools to extract information about town committees and their meeting documents. You have access to tools, but only use them when necessary. If a tool is not required, respond as normal.",
                messages=initial_messages,
                tools=[tool.get_tool_definition() for tool in self.tools.values()],
                tool_choice={"type": "auto"}
            )

            print(f"Original results: {response}")
            
            # Process the message and handle tool calls
            result = self.handle_tool_calls(response, initial_messages)
            
            # Add metadata to result
            result["town"] = town_name
            if state:
                result["state"] = state
            result["website"] = website_url
            
            return result
            
        except Exception as e:
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    import os
    
    # Create analyzer and run it
    analyzer = TownWebsiteAnalyzer()
    result = analyzer.run_workflow("Williston", "VT")
    
    # Save results to file
    with open("cambridge_committees.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Analysis complete. Found {len(result.get('committees', []))} committees.")