from unittest.mock import AsyncMock, patch

import pytest
from bs4 import BeautifulSoup

from tools.site_scraper import Bs4SiteScraperTool


@pytest.mark.asyncio
async def test_execute_with_extractions():
    # Create the tool instance
    scraper = Bs4SiteScraperTool()

    # Test with real website and parameters
    result = await scraper.execute({
        "url": "http://willistonvt.govoffice3.com/index.asp?Type=B_DIR&SEC=%7B3076BCA1-3425-474F-8F7E-1103631082A0%7D",
        "extract_links": ["agenda", "minutes", "meeting"],
        "extract_body_text": True,
        "extract_navigation": True
    })

    
    assert result["url"] == "http://willistonvt.govoffice3.com/index.asp?Type=B_DIR&SEC=%7B3076BCA1-3425-474F-8F7E-1103631082A0%7D"
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
    
    # Check navigation
    assert "navigation" in result
    assert len(result["navigation"]) > 0