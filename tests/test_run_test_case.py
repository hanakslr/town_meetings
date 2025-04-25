import pytest
from strategies.save import save_fetching_strategy, save_params
from tools.iterate_strategy import run_test_case

fetching_strategy = {
    "committee_name": "Community Forest Committee",
    "strategy_type": "yearly_archive",
    "strategy_name": "yearly_meeting_pages_with_pdf_links",
    "schema": {
        "base_url": "Base URL of the town website",
        "year_links_url": "URL to the committee's meeting page with links to yearly meeting pages",
        "year_pattern": "Pattern to identify yearly meeting pages from links",
        "meeting_date_selector": "CSS selector to identify individual meeting date links on yearly pages",
        "pdf_link_pattern": "Pattern to identify agenda PDF links on meeting detail pages",
    },
    "values": {
        "base_url": "https://www.anytown.us",
        "year_links_url": "https://www.anytown.us/index.asp?Type=B_BASIC&SEC={73312477-3405-4A85-BD03-685A9EC64AC6}",
        "year_pattern": "Community Forest Committee (\\d{4}) Meetings",
        "meeting_date_selector": "a[href*='SEC'][href*='DE']",
        "pdf_link_pattern": "(?i)agenda.*\\.pdf|(?i)meeting.*\\.pdf",
    },
    "notes": "The committee organizes meeting information by year, with each year having its own page. Each meeting date has its own detail page containing PDF links to agendas and minutes. Agendas may be named using variations including 'agenda' or other meeting documentation.",
    "code": "import re\nfrom bs4 import BeautifulSoup\nimport requests\n\ndef get_committee_agendas(base_url, year_links_url, year_pattern, meeting_date_selector, pdf_link_pattern):\n    # Get yearly meeting pages\n    response = requests.get(year_links_url)\n    soup = BeautifulSoup(response.text, 'html.parser')\n    \n    # Find links to yearly meeting pages\n    year_links = []\n    for link in soup.find_all('a'):\n        if link.text and re.search(year_pattern, link.text):\n            year_links.append({'year': re.search(year_pattern, link.text).group(1), \n                               'url': base_url + '/' + link['href'] if link['href'].startswith('index') else link['href']})\n    \n    all_agendas = []\n    \n    # For each year page, get meeting date links\n    for year_link in year_links:\n        response = requests.get(year_link['url'])\n        year_soup = BeautifulSoup(response.text, 'html.parser')\n        \n        # Find meeting date links\n        meeting_links = year_soup.select(meeting_date_selector)\n        \n        # For each meeting, get agenda PDFs\n        for meeting in meeting_links:\n            meeting_url = base_url + '/' + meeting['href'] if meeting['href'].startswith('index') else meeting['href']\n            meeting_date = meeting.text.strip()\n            \n            # Get meeting detail page\n            meeting_response = requests.get(meeting_url)\n            meeting_soup = BeautifulSoup(meeting_response.text, 'html.parser')\n            \n            # Find PDF links that match agenda pattern\n            for pdf_link in meeting_soup.find_all('a'):\n                if pdf_link.get('href') and pdf_link.get('href').endswith('.pdf'):\n                    link_text = pdf_link.text.strip()\n                    if re.search(pdf_link_pattern, link_text) or re.search(pdf_link_pattern, pdf_link['href']):\n                        agenda_url = base_url + pdf_link['href'] if not pdf_link['href'].startswith('http') else pdf_link['href']\n                        all_agendas.append({\n                            'committee': 'Community Forest Committee',\n                            'year': year_link['year'],\n                            'meeting_date': meeting_date,\n                            'agenda_url': agenda_url,\n                            'agenda_title': link_text\n                        })\n    \n    return all_agendas",
}


async def test_run_test_case():
    results = await run_test_case(
        fetching_strategy["code"], fetching_strategy["values"], {}
    )

    print(results)


## These are just for local testing to avoid calling the LLM and running through the whole workflow when not needed
@pytest.mark.skip
async def test_dump_strategy():
    save_fetching_strategy(fetching_strategy)


@pytest.mark.skip
async def test_dump_params():
    save_params("fake_committee", fetching_strategy)
