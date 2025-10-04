import requests
from bs4 import BeautifulSoup
from typing import Dict, List
from urllib.parse import quote

def get_interactions(drug: str) -> Dict[str, List[str]]:
    encoded_drug = quote(drug.lower())
    drug_interactions_url = f"https://www.drugs.com/drug-interactions/{encoded_drug}-index.html?filter=3"
    
    return _scrape_interactions(drug_interactions_url)

def _scrape_interactions(url: str) -> List[str]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Priority': 'u=0, i',
            'TE': 'trailers'
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        # Add some basic cookies that might be needed
        cookies = {
            'ddc-detect-cookie': '1',
            'ddc-pvc': '17'
        }
        
        response = session.get(url, cookies=cookies, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        interactions = []
        
        # Look for drug interaction lists with class 'interactions'
        interaction_lists = soup.find_all('ul', class_='interactions')
        for ul in interaction_lists:
            li_items = ul.find_all('li')
            for li in li_items:
                # Get the drug name from the link text
                link = li.find('a')
                if link:
                    drug_name = link.get_text(strip=True)
                    if drug_name:
                        interactions.append(drug_name)

        interactions = [i for i in interactions if 'moderate' not in i.lower() and 'minor' not in i.lower() and 'major' not in i.lower()]
        return interactions
        
    except Exception as e:
        print(f"Error scraping drug interactions: {e}")
        return []