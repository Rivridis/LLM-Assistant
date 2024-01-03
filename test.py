from bs4 import BeautifulSoup
import os
import requests

header = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'}

from duckduckgo_search import DDGS
link = ""
with DDGS() as ddgs:
    for r in ddgs.text('good chocolate cake reciepe', region='in-en', safesearch='off', timelimit='y', max_results=1):
        link = r["href"]
        print(link)

resp = requests.get(link,headers=header)
html = resp.text
soup = BeautifulSoup(html, "html.parser")
text = soup.body.get_text().strip()
text = os.linesep.join([s for s in text.splitlines() if s])
print(text)