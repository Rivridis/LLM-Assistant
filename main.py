# Imports
from llama_cpp import Llama,LlamaGrammar
import requests
import re
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
import gradio as gr

# Random User Agent
software_names = [SoftwareName.CHROME.value]
operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]   
user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)
user_agent = user_agent_rotator.get_random_user_agent()

# GBNF Grammar
grmtxt = r'''
root ::= answer
answer ::= "{"   ws   "\"thought\":"   ws   string   ","   ws   "\"tool\":"   ws   stringlist   "}"
answerlist ::= "[]" | "["   ws   answer   (","   ws   answer)*   "]"
string ::= "\""   ([^"]*)   "\""
boolean ::= "true" | "false"
ws ::= [ \t\n]*
number ::= [0-9]+   "."?   [0-9]*
stringlist ::= "["   ws   "]" | "["   ws   string   (","   ws   string)*   ws   "]"
numberlist ::= "["   ws   "]" | "["   ws   string   (","   ws   number)*   ws   "]"
'''
grammar = LlamaGrammar.from_string(grmtxt)

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
    You are an AI model that is trained in tool and function calling. Think through the question, and return what all functions to call and how to call them, to give the best response to user's question.
    These are modules available to you. Make sure to call the correct function, and respond as given in the output example.
    
    google(query): This function is used only when you can't answer a question, and need factual details.
    Example:
    input : How old is the moon?
    output : {"thought":"I need to google this query", "tool": ["google(how old is the moon)"]}

    date_loc(): This function is used when the user needs details about weather, date, time, basically if you need location data for an answer.
    Example:
    input: Where am I located, and how cold is it here today?
    output : {"thought":"I need the location for this question", "tool": ["date_loc()"]}

    MUSIC MODULE
    music_play(name): This function is used when the user needs details about music currently playing in their system, or its status.
    music_pause(): This function is used to pause music
    music_get(): This function is used to get currently playing music.
    Example:
    input: play never gonna give you up
    output : {"thought":"I need to play music", "tool": ["music_play(never gonna give you up)"]}

    input: what music is playing?
    output : {"thought":"I need to check current music", "tool": ["music_get()"]}

    Mail Module
    mail_get(): This function is used when the user wants to read their mails
    Example:
    input: read my mails please
    output : {"thought":"I need use the mail module", "tool": ["mail()"]}
    
    calc_no(query): This function is used to calculate numbers.
    Example:
    input: what is 23*657/345
    output: {"thought":"I need to calculate this value", "tool": ["calc_no(23*657/345)"]}
    
    none(): This function is used when you feel that the user is just chatting with the language model. Also use this function to refer back to previous conversations or questions.
    Example:
    input: Hello! how are you today?
    output: {"thought":"The user is just chatting with me", "tool": ["none()"]}
    input: Please provide me the reciepe for the cake you mentioned earlier
    output: {"thought":"The user wants to refer back to chat history", "tool": ["none()"]}
    input: Continue
    output: {"thought":"The user wants to refer back to chat history", "tool": ["none()"]}
    
    end(): This function is called when the LLM feels the user wants to end the conversation.
    Example:
    input: I want to go to bed, goodnight!
    output: {"thought":"The user wants to end conversation", "tool": ["end()"]}

    Example:
    input: what music is playing on my system? and what is 122-7?
    output: {"thought":"I need to call multiple modules", "tool": ["music_get()","calc_no(122-7)"]}

    Example:
    input: reccomend me a good music, and play it
    output: {"thought":"reccomending a music for you to play", "tool": ["music_play(Sparkle kimi no na wa)"]}
    
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
    grammar=grammar
    )
    search_dict = output['choices'][0]['text']
    return(search_dict)
    
    # Normal Chatting
    if search_dict['tool'] == "none()":
        system2 = """You are an AI chat assistant named Luna, trained to help the user with any of their questions, and have a nice friendly chat with them. You are provided with a summarized chat history, which you can use to refer back to conversations.
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
        chat_memory+="user: {}\nassistant: {}\n".format(prompt,output['choices'][0]['text'])
        
        # Chat History Summarising
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
        return(output['choices'][0]['text'])
    
    # Number Calculation
    elif "calc_no" in search_dict['tool']:
        pattern = r'\(([^)]+)\)'
        match = re.search(pattern, str(search_dict['tool']))
        if match:
          numeric_equation = str(match.group(1))
          return(str(eval(numeric_equation)))
    
    # Internet Search
    elif "google" in search_dict["tool"]:
        system4 = """You are an AI chat assistant named Luna, trained to help the user with any of their questions, having connection to the internet. You are provided with the search result of the google search of user's query below. Use the information to formulate the best response to the user's query. Go through the paragraphs carefully.
        """
        link = ""
        mainp = ''
        
        pattern = r"\(([^)]+)\)"
        matches = re.findall(pattern, str(search_dict['tool']))
        
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
            system4+= mainp
        
        else:
            system4 += mainp
        
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
        chat_memory+="user: {}\nassistant: {}\n".format(prompt,output['choices'][0]['text'])
        return(output['choices'][0]['text'])
    
    else:
       return "See you later!"
            
    llm.reset()
    return()

# Main Code
gr.ChatInterface(chat,
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Enter Question", container=False, scale=7),
    title="AI Assistant",
    theme="soft",
    examples=["Good Morning!", "Google en passant", "what is 899*99/21"],
    clear_btn="Clear",).launch()
