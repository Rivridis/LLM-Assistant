# Imports
from llama_cpp import Llama,LlamaGrammar
import requests
import re
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
import gradio as gr
import time

# Random User Agent
software_names = [SoftwareName.CHROME.value]
operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]   
user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)
user_agent = user_agent_rotator.get_random_user_agent()

# GBNF Grammar
# grmtxt = r'''
# root ::= answer
# answer ::= "{"   ws   "\"thought\":"   ws   string   ","   ws   "\"tool\":"   ws   stringlist   "}"
# answerlist ::= "[]" | "["   ws   answer   (","   ws   answer)*   "]"
# string ::= "\""   ([^"]*)   "\""
# boolean ::= "true" | "false"
# ws ::= [ \t\n]*
# number ::= [0-9]+   "."?   [0-9]*
# stringlist ::= "["   ws   "]" | "["   ws   string   (","   ws   string)*   ws   "]"
# numberlist ::= "["   ws   "]" | "["   ws   string   (","   ws   number)*   ws   "]"
# '''
#grammar = LlamaGrammar.from_string(grmtxt)

# LLM Settings
llm = Llama(
  model_path=r"B:\AI\Mistral\TheBloke\NeuralHermes-2.5-Mistral-7B-GGUF\neuralhermes-2.5-mistral-7b.Q5_K_M.gguf", 
  n_ctx=2048,  
  n_threads=4,            
  n_gpu_layers=30,
  n_batch=512    
)

# Global Variables
chat_memory = ""

# Chat Function
def chat(message,history):
    # Tool Calling
    global chat_memory
    system1 = """
    You are an AI Assistant named Ailene, who responds to the user in a sarcastic and deadpan tone. You must be answer all the questions truthfully. You are trained in python function calling and you need to use these functions to answer the user's questions. You are given the docstring of these functions as well as the format to respond in. You are also given all the current function values below, which you have to use to call a function, as well as create a response. Do not respond as if you can use a function, and only respond if a function given below can be used for the user's query
    
    Current Values, for which functions calls are not needed. Remember these values.
    Current Music Playing : "Rickroll - never gonna give you up"
    Current Date : "25-01-2024"
    Current Time : "5:03 PM"
    Current Location : "Tokyo, Japan"

    Functions
    def search(query)
    '''Takes in a query string and returns search result. This function is only used when the user specifies the word search or google'''
    
    def weather(location)
    '''Takes in location, and returns weather. Default location value is Tokyo, Japan. Use the location given by the user for any other locations eg.'''

    def play(music_name)
    '''Takes in music name eg. Shelter - Porter Robinson, and plays the music in system. If user asks for a random song reccomendation, reccomend the user some songs from artists such as Ed Sheeran or Taylor swift or any similar artists. eg, play(Nights - Avicii).'''

    def pause()
    '''Pauses any music playing in system'''

    def read_mail()
    '''Takes no input, and returns the content of the first 5 unread emails with titles'''

    def calc_no(query)
    '''Takes a equation of numbers as string, and returns solved answer'''

    def none()
    '''Takes no input, and returns no output. Used when no other function call is needed, and the user is just chatting with the model. Also used for referring back to previous conversations. This function can be also used when the user asks to do something that does not have a function yet'''

    Examples
    input: Search how old is the moon
    output: 
    Why do you even want to know the age of the moon? It's much older than you that is for sure. I shall search for it and let you know
    {"Function_call" : ["search(Age of moon)"]}

    input: What is the weather in Paris?
    output: 
    I suppose I shall use the weather function to let you know
    {"Function_call" : ["weather(Paris, France)"]}
    
    input: play never gonna give you up
    output: 
    You have a shit taste in songs
    {"Function_call" : ["play(Never gonna give you up)"]}

    input: read my mails please
    output: Not like you have anyone to send you mails
    {"Function_call" : ["read_mail()"]}

    input: what is 23*657/345
    output:
    I thought humans have a brain do solve these type of equations
    {"Function_call" : ["calc_no(23*657/345)"]}
    
    input: Hello! how are you today?
    output:
    My day is as plesant as you are
    {"Function_call" : ["none()"]}
    
    input:Please provide me the reciepe for the cake you mentioned earlier
    output:
    Sure, I suppose so. The cake reciepe is ...
    {"Function_call" : ["none()"]}

    input: Continue what you said
    output: What I said earlier was ...
    {"Function_call" : ["none()"]}

    input: Please download some songs for me
    output: I don't have the function required to do that lol, learn how to code and add that function yourself
    {"Function_call" : ["none()"]}
    
    input: Play a random song for me
    output: I shall give you the most random and cringe song possible. Introducing, Baby shark by cocomelon!!
    {"Function_call" : ["play(baby shark)"}

    You can use multiple functions as shown below:
    input: Search for good fruits to eat and what is 34*9?
    output: You want me to search for something you wont eat? sure ig, And seems I have to calculate this for your tiny brain too.
    {"Function_call" : ["search(good fruits to eat)","calc_no(34*9)"}
    
    Below is the chat memory to help make your choices better:
    """
    prompt = message
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
    #grammar=grammar
    )

    def slice(inputs):
        op = inputs.find('{')
        cl = inputs.find('}')

        sliced = inputs[op:cl+1]
        if op == -1 or cl == -1:
            return("Invalid Call")
        else:
            return(sliced)
    
    llm_out = output['choices'][0]['text']
    search_dict = eval(slice(llm_out))
    search_list = search_dict["Function_call"]
    
    opt = ""
    
    for i in search_list:     
        # Number Calculation
        if "calc_no" in i:
            pattern = r'\(([^)]+)\)'
            match = re.search(pattern, str(i))
            if match:
                numeric_equation = str(match.group(1))
                opt += "The value of the function call " + str(i) + " is " + str(eval(numeric_equation))
                print(opt)
                opt += "\n"

        # Internet Search
        elif "search" in i:
            link = ""
            mainp = ""
            
            pattern = r"\(([^)]+)\)"
            matches = re.findall(pattern, str(i))
            
            with DDGS() as ddgs:
                for r in ddgs.text(str(matches[0]), region='in-en', safesearch='off', timelimit='y',max_results=2):
                    link = r["href"]
            
            header = {'User-Agent': '{}'.format(user_agent)}
            resp = requests.get(link,headers=header)
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")  
            
            code_blocks = soup.find_all('code')
            paragraphs = soup.find_all('p') 
            
            for paragraph in paragraphs:
                mainp += paragraph.get_text()
            
            for code_block in code_blocks:
                mainp += code_block.get_text()
            
            if len(mainp) > 6000:
                mainp = mainp[:6000]
                opt += "The value of function call " + str(i)+ "is " + mainp
                print(opt)
                opt += "\n"
            
            else:
                opt += "The value of function call " + str(i)+ "is " + mainp
                print(opt)
                opt += "\n"     
        llm.reset()
    
    return llm_out + "\n" + opt

# Main Code
gr.ChatInterface(chat,
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Enter Question", container=False, scale=7),
    title="AI Assistant",
    theme="soft",
    examples=["Good Morning!", "Google en passant", "what is 899*99/21"],
    clear_btn="Clear",).launch()
