from llama_cpp import Llama,LlamaGrammar

grmtxt = r'''
root ::= Reply
Reply ::= "{" "\"Value\":" outval   "}"
Replylist ::= "[]" | "["   ws   Reply   (","   ws   Reply)*   "]"
outval ::= "\"" "Positive" "\"" | "\"" "Negative" "\"" | "\"" "Neutral" "\""
string ::= "\""   ([^"]*)   "\""
boolean ::= "true" | "false"
ws ::= [ \t\n]*
number ::= [0-9]+   "."?   [0-9]*
stringlist ::= "["   ws   "]" | "["   ws   string   (","   ws   string)*   ws   "]"
numberlist ::= "["   ws   "]" | "["   ws   string   (","   ws   number)*   ws   "]"

'''
llm = Llama(
  model_path=r"model\neuralhermes-2.5-mistral-7b.Q5_K_M.gguf", 
  n_ctx=2048,  
  n_threads=4,            
  n_gpu_layers=30,
  n_batch=512    
)
grammar = LlamaGrammar.from_string(grmtxt)
system1 = r"""
You are an Stock AI, who analyzes given stock news data, and classifies these news as positive, negative or neutral stock trends.

Input
Rs 14 lakh-crore shock! Sensex crashes 900 points. Is the smallcap bubble bursting?
Sensex on Wednesday ended 906 points to crack below the 73,000-level while Nifty also fell below 22,000 but the pain was unbearable on the other end of the market comprising smaller stocks. The smallcap index fell over 5% to record the worst single-day fall since December 2022, midcaps lost 4% while microcaps and SME stock indices fell around 6% each as the stellar rally in the broader market is seen taking a pause.

Output
{"Value":"Negative"}

Input
ITC shares surge 8% following BAT's 3.5% stake sale; biggest intraday gain in 4 years
The stake sale could lead to a re-rating of ITC's stock, per experts and the increase in free float will cause an upward adjustment in the weights on the index.

Output
{"Value":"Positive"}


The news article is given below.
"""

prompt = r'''Macrotech Developers shares fall 3% after promoter sells stake'''
output = llm(
"<|im_start|>system {}<|im_end|>\n<|im_start|>user {}<|im_end|>\n<|im_start|>assistant".format(system1,prompt),
max_tokens=-1,
stop=["<|im_start|>","<|im_end|>"],
temperature=0.8,
top_k=40,
repeat_penalty=1.1,
min_p=0.05,
top_p=0.95,
echo=False,
grammar=grammar
)

llm_out = output['choices'][0]['text']
print(llm_out)