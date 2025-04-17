import requests
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class TownMeetingDiscovery:
    def __init__(self):
        """Initialize the workflow with your Anthropic API key."""
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
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
        
        # Basic validation
        if not website_url.startswith('http'):
            website_url = 'https://' + website_url
            
        # Verify the website is accessible
        try:
            response = self.session.get(website_url, timeout=10)
            if response.status_code != 200:
                raise Exception(f"Could not access {website_url}, status code: {response.status_code}")
        except Exception as e:
            raise Exception(f"Error accessing town website: {str(e)}")
            
        return website_url

    def run_workflow(self, town_name, state=None):
        # Step 1: Find town website
        print(f"Finding official website for {town_name}...")
        website_url = self.find_town_website(town_name, state)
        return website_url



if __name__ == "__main__":
    import os
       
    # Create workflow and run it
    workflow = TownMeetingDiscovery()
    result = workflow.run_workflow("Cambridge", "MA")
    print(result)