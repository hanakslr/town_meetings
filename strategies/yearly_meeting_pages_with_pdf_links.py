class YearlyMeetingPagesWithPdfLinks(FetchingStrategy):
    name = "yearly_meeting_pages_with_pdf_links"
    
    def get_committee_agendas(self, base_url, year_links_url, year_pattern, meeting_date_selector, pdf_link_pattern):
        # Get yearly meeting pages
        response = requests.get(year_links_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find links to yearly meeting pages
        year_links = []
        for link in soup.find_all('a'):
            if link.text and re.search(year_pattern, link.text):
                year_links.append({'year': re.search(year_pattern, link.text).group(1), 
                                   'url': base_url + '/' + link['href'] if link['href'].startswith('index') else link['href']})
        
        all_agendas = []
        
        # For each year page, get meeting date links
        for year_link in year_links:
            response = requests.get(year_link['url'])
            year_soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find meeting date links
            meeting_links = year_soup.select(meeting_date_selector)
            
            # For each meeting, get agenda PDFs
            for meeting in meeting_links:
                meeting_url = base_url + '/' + meeting['href'] if meeting['href'].startswith('index') else meeting['href']
                meeting_date = meeting.text.strip()
                
                # Get meeting detail page
                meeting_response = requests.get(meeting_url)
                meeting_soup = BeautifulSoup(meeting_response.text, 'html.parser')
                
                # Find PDF links that match agenda pattern
                for pdf_link in meeting_soup.find_all('a'):
                    if pdf_link.get('href') and pdf_link.get('href').endswith('.pdf'):
                        link_text = pdf_link.text.strip()
                        if re.search(pdf_link_pattern, link_text) or re.search(pdf_link_pattern, pdf_link['href']):
                            agenda_url = base_url + pdf_link['href'] if not pdf_link['href'].startswith('http') else pdf_link['href']
                            all_agendas.append({
                                'committee': 'Catamount Community Forest Committee',
                                'year': year_link['year'],
                                'meeting_date': meeting_date,
                                'agenda_url': agenda_url,
                                'agenda_title': link_text
                            })
        
        return all_agendas
from strategies import FetchingStrategy
import re
from bs4 import BeautifulSoup
import requests