import requests
from bs4 import BeautifulSoup

def extract_links(url, domain):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True) if domain in a['href']]
    return links

def crawl_website(start_url, domain):
    visited = set()
    to_visit = [start_url]
    
    while to_visit:
        current_url = to_visit.pop()
        if current_url not in visited:
            visited.add(current_url)
            print(f"Crawling: {current_url}")
            
            # Extract links and add them to the to_visit list
            links = extract_links(current_url, domain)
            to_visit.extend(links)
    
    return visited