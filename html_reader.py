import requests
from bs4 import BeautifulSoup

class HTMLReader():
    def __init__(self):
        pass

    def find_player_url(self, team):
        dynamic_url = f"https://www.bcci.tv/international/{team.lower()}/players"
        player_urls = [] 
        player_names = []

        req = requests.get(dynamic_url)
        soup = BeautifulSoup(req.text, "html.parser")
        links = soup.find_all("a")    

        for link in links:
            href = link.get("href")
            if dynamic_url + "/" in href:
                player_urls.append(href)
                player_names.append(self.get_player_name(href))
        
        return player_urls, player_names
    
    def get_player_name(self, url):
        try:
            return ' '.join(url.split("/")[-2].split('-')).title()
        except Exception:
            return " "
