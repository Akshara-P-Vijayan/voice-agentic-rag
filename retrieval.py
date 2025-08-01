import wikipedia, arxiv, requests
from bs4 import BeautifulSoup

DUCK_API = "https://api.duckduckgo.com/?q={query}&format=json"

def fetch_wikipedia(q):
    try: return wikipedia.summary(q, sentences=2)
    except: return ""

def fetch_arxiv(q):
    try:
        res = next(arxiv.Search(query=q, max_results=1).results(), None)
        return res.summary if res else ""
    except: return ""

def fetch_duckduckgo(q):
    try:
        data = requests.get(DUCK_API.format(query=q)).json()
        return data.get("Abstract", "")
    except: return ""

def fetch_context(query):
    return fetch_wikipedia(query) or fetch_arxiv(query) or fetch_duckduckgo(query)
