from llama_cpp import Llama,LlamaGrammar
import requests
from googlesearch import search
import re
from bs4 import BeautifulSoup
import os
from duckduckgo_search import DDGS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

grmtxt = r'''
root ::= answer
answer ::= "{"   ws   "\"thought\":"   ws   string   ","   ws   "\"tool\":"   ws   string   "}"
answerlist ::= "[]" | "["   ws   answer   (","   ws   answer)*   "]"
string ::= "\""   ([^"]*)   "\""
boolean ::= "true" | "false"
ws ::= [ \t\n]*
number ::= [0-9]+   "."?   [0-9]*
stringlist ::= "["   ws   "]" | "["   ws   string   (","   ws   string)*   ws   "]"
numberlist ::= "["   ws   "]" | "["   ws   string   (","   ws   number)*   ws   "]"
'''
grammar = LlamaGrammar.from_string(grmtxt)
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path=r"B:\AI\Mistral\TheBloke\NeuralHermes-2.5-Mistral-7B-GGUF\neuralhermes-2.5-mistral-7b.Q5_K_M.gguf", 
  n_ctx=2048,  
  n_threads=4,            
  n_gpu_layers=30,
  n_batch=512    
)
chat_memory = ""

while True:
    system1 = """
    You are an AI model that is trained in tool and function calling. Think through the question, and return what all functions to call and how to call them, to give the best response to user's question.
    These are the python functions avaliable to you. Make sure to call the correct function, and respond as given in the output example.
    
    google(query): This function is used only when you can't answer a question, and need factual details.
    input : How old is the moon?
    output : {"thought":"I need to google this query", "tool": "google(how old is the moon)"}

    location(): This function is used when the user needs details about weather, date, time, basically if you need location data for an answer.
    input: Where am I located, and how cold is it here today?
    output : {"thought":"I need the location for this question", "tool": "location()"}
    
    calc_no(query): This function is used to calculate numbers.
    input: what is 23*657/345
    output: {"thought":"I need to calculate this value", "tool": "calc_no(23*657/345)"}
    
    none(): This function is used when you feel that the user is just chatting with the language model. Also use this function to refer back to previous conversations or questions.
    input: Hello! how are you today?
    output: {"thought":"The user is just chatting with me", "tool": "none()"}
    input: Please provide me the reciepe for the cake you mentioned earlier
    output: {"thought":"The user wants to refer back to chat history", "tool": "none()"}
    
    end(): This function is called when the LLM feels the user wants to end the conversation.
    input: I want to go to bed, goodnight!
    output: {"thought":"The user wants to end conversation", "tool": "end()"}

    Below is the chat memory to help make your choices better:
    """
    prompt = input("Enter question: ")
    output = llm(
    "<|im_start|>system {}<|im_end|>\n<|im_start|>user {}<|im_end|>\n<|im_start|>assistant".format(system1+chat_memory,prompt),
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
    search_dict = eval(output['choices'][0]['text'])
    print(search_dict)
    print(chat_memory)
    
    if search_dict['tool'] == "none()":
        system2 = """You are an AI chat assistant named LUNA, trained to help the user with any of their questions, and have a nice friendly chat with them. You are provided with a summarized chat history, which you can use to refer back to conversations.
        """
        output = llm(
        "<|im_start|>system {}<|im_end|>\n<|im_start|>user {}<|im_end|>\n<|im_start|>assistant".format(system2+chat_memory,prompt),
        max_tokens=-1,
        stop=["<|im_start|>","<|im_end|>"],
        temperature=0.8,
        top_k=40,
        repeat_penalty=1.1,
        min_p=0.05,
        top_p=0.95,
        echo=False
        )
        print(output['choices'][0]['text'])

        chat_memory+="user: {}\nassistant: {}\n".format(prompt,output['choices'][0]['text'])
        
        if len(chat_memory) > 6000:
            system3 = """You are an AI that summarises the given chat history to the least words possible, while extracting and preserving key information.
            """
            output = llm(
            "<|im_start|>system {}<|im_end|>\n<|im_start|>user {}<|im_end|>\n<|im_start|>assistant".format(system3,chat_memory),
            max_tokens=-1,
            stop=["<|im_start|>","<|im_end|>"],
            temperature=0.8,
            top_k=40,
            repeat_penalty=1.1,
            min_p=0.05,
            top_p=0.95,
            echo=False
            )
            chat_memory = output['choices'][0]['text']
    
    elif "calc_no" in search_dict['tool']:
        pattern = r'\(([^)]+)\)'
        match = re.search(pattern, str(search_dict['tool']))
        if match:
          numeric_equation = str(match.group(1))
          print(eval(numeric_equation))
    
    elif "google" in search_dict["tool"]:
        system4 = """You are an AI chat assistant named LUNA, trained to help the user with any of their questions, having connection to the internet. You are provided with the search result of the google search of user's query below. Use the information to formulate the best response to the user's query. Go through the paragraphs carefully.
        """
        pattern = r"\(([^)]+)\)"
        matches = re.findall(pattern, str(search_dict['tool']))
        url = search(str(matches[0]), num_results=1)
        link = ""
        with DDGS() as ddgs:
            for r in ddgs.text(str(matches[0]), region='in-en', safesearch='off', timelimit='y',max_results=2):
                link = r["href"]
        header = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'}
        resp = requests.get(link,headers=header)
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all('p')
        mainp = ''
        for paragraph in paragraphs:
            mainp += paragraph.get_text()
        if len(mainp) > 6000:
            mainp = mainp[:6000]
            system4+= mainp
        
        else:
            system4 += mainp
        print(mainp)
        
        output = llm(
        "<|im_start|>system {}<|im_end|>\n<|im_start|>user {}<|im_end|>\n<|im_start|>assistant".format(system4,prompt),
        max_tokens=-1,
        stop=["<|im_start|>","<|im_end|>"],
        temperature=0.8,
        top_k=40,
        repeat_penalty=1.1,
        min_p=0.05,
        top_p=0.95,
        echo=False
        )
        print(output['choices'][0]['text'])
        chat_memory+="user: {}\nassistant: {}\n".format(prompt,output['choices'][0]['text'])
    else:
       print("Cyaa!")
       break
            
    llm.reset()