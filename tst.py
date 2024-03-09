from duckduckgo_search import DDGS

results = DDGS().news( keywords="India news",region="in-en", safesearch="off", timelimit="m", max_results=20)
print(results)