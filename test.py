from duckduckgo_search import DDGS

results = DDGS().videos(
    keywords="cat videos",
    region="wt-wt",
    safesearch="off",
    timelimit="w",
    resolution="high",
    duration="medium",
    max_results=10,
)

val = ""
for i in results:
    val += str(i['content'])
    val += '\n'
    val += str(i['description'])
    val += '\n'


print(val)