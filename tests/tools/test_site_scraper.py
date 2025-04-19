import json
import os

import pytest

# This isn't scalable but works for now.
from dotenv import load_dotenv

from tools.site_scraper import Bs4SiteScraperTool

load_dotenv()

@pytest.mark.asyncio
async def test_text_body_extraction():
    """This is more of a test for isolating the site scraper instead of a strict unit test."""

    # Create the tool instance
    scraper = Bs4SiteScraperTool()

    # Test with real website and parameters
    result = await scraper.execute({
        "url": os.environ["TEST_WEBSITE_URL_1"],
        "extract_links": ["agenda", "minutes", "meeting"],
        "extract_body_text": True
    })

    print(json.dumps(result, indent=2))

    assert result["title"] is not None
    
    # Check links
    assert "links" in result
    assert len(result["links"]) > 0
    # Verify at least some links contain the keywords we're looking for
    link_texts = [link["text"].lower() for link in result["links"]]
    assert any(any(keyword in text for keyword in ["agenda", "minutes", "meeting"]) for text in link_texts)
    
    # Check main text
    assert "main_text" in result
    assert len(result["main_text"]) > 0