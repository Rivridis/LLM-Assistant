from llama_cpp import Llama
from duckduckgo_search import DDGS

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path=r"B:\AI\Mistral\TheBloke\dolphin-2.2.1-mistral-7b.Q5_K_M.gguf\dolphin-2.2.1-mistral-7b.Q5_K_M.gguf", 
  n_ctx=2048,  
  n_threads=4,            
  n_gpu_layers=35,
  n_batch=512    
)

system1 = """You are an AI model that is trained for giving search topics to user's questions. give 3 search topics that might answer the user's queries in a strict list format. You do not respond to the user query directly, and you follow the formatting. searches should be made up of only sentences.

["search1","search2",search3"]
"""
prompt = "when was narendera modi born?"

output = llm(
  "<|im_start|>system {}<|im_end|>\n<|im_start|>user {}<|im_end|>\n<|im_start|>assistant".format(system1,prompt),
  max_tokens=-1,
  stop=["<|im_start|>","<|im_end|>"],
  temperature=0.8,
  top_k=40,
  repeat_penalty=1.1,
  min_p=0.05,
  top_p=0.95,
  echo=False
)
search_list = eval(output['choices'][0]['text'])
rlist = []
with DDGS() as ddgs:
    for r in ddgs.text(search_list[0],max_results=2):
            rlist.append(r)
rfinlist = []
rfinlist.append(rlist[0]['body'])
rfinlist.append(rlist[1]['body'])

system2="""using the google search results, formulate the best response for the user.
Results:{},{}
""".format(rfinlist[0],rfinlist[1])
print(system2)

output = llm(
  "<|im_start|>system {}<|im_end|>\n<|im_start|>user {}<|im_end|>\n<|im_start|>assistant".format(system2,prompt),
  max_tokens=300,
  stop=["<|im_start|>","<|im_end|>"],
  temperature=0.8,
  top_k=40,
  repeat_penalty=1.1,
  min_p=0.05,
  top_p=0.95,
  echo=False,
  stream=True,

)
print("\n")
print("Final Answer")
for token in output:
    print(token["choices"][0]["text"],end="")