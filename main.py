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

#GBNF Grammar
grmtxt = r'''
root ::= output
output ::= "{" ws "\"assistant_reply\":" ws string "," ws "\"function_called\":" ws functionvalueslist "}"
outputlist ::= "[]" | "[" ws output ("," ws output)* "]"
functionvalues ::=  "\"" "play" "(" functionparameter ")" "\"" | "\"none()\"" | "\"" "search" "(" functionparameter ")" "\"" | "\"" "weather" "(" functionparameter ")" "\"" | "\"" "pause" "(" functionparameter ")" "\"" | "\"" "read_mail" "(" functionparameter ")" "\"" | "\"" "calc_no" "(" functionparameter ")" "\""
functionvalueslist ::= "[" ws functionvalues ("," ws functionvalues)* ws "]"
functionparameter ::= ([^"]*)
string ::= "\"" ([^"]*) "\""
boolean ::= "true" | "false"
ws ::= [ \t\n]*
number ::= [0-9]+  "."?  [0-9]*
stringlist ::= "[" ws "]" | "[" ws string ("," ws string)* ws "]"
'''
grammar = LlamaGrammar.from_string(grmtxt)

# LLM Settings
llm = Llama(
  model_path=r"model\neuralhermes-2.5-mistral-7b.Q5_K_M.gguf", 
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
    You are an AI Assistant named Vivy, who responds to the user with helpful information, tips, and jokes just like Jarvis from the marvel universe. You must be answer all the questions truthfully. You are trained in python function calling and you need to use these functions to answer the user's questions. You are given the docstring of these functions as well as the format to respond in. You are also given all the current function values below, which you have to use to call a function, as well as create a response. Do not respond as if you can use a function, and only respond if a function given below can be used for the user's query. Ask the user for more details before calling a function, and use none() when function call is not needed. Make sure to call functions when necessary, according to the given context in the question.
    
    Current Values, for which functions calls are not needed. Remember these values.
    Current Music Playing : "Never gonna give you up"
    Current Date : "25-01-2024"
    Current Time : "5:03 PM"
    Current Location : "Tokyo, Japan"

    Functions
    def search(query)
    '''Takes in a query string and returns search result. Use this function, when a user's query needs some information from the internet. Feel free to use it whenever you feel like it.'''
    
    def weather(location)
    '''Takes in location, and returns weather. Default location value is Tokyo, Japan. Use the location given by the user for any other locations eg. This function is used for retrieving weather data, temperature, pressure etc when the user asks for it.''

    def play(music_name)
    '''Takes in music name eg. Shelter - Porter Robinson, and plays the music in system. If user asks for a random song reccomendation, reccomend the user some songs from artists such as Ed Sheeran or Taylor swift or any similar artists. eg, play(Nights - Avicii). Always reccomend the user a song, and don't give a general name'''

    def pause()
    '''Pauses any music playing in system'''

    def read_mail()
    '''Takes no input, and returns the content of the first 5 unread emails with titles'''
    
    def calc_no(query)
    '''Takes a equation of numbers and returns solved answer'''

    def none()
    '''Takes no input, and returns no output. Used when no other function call is needed, and the user is just chatting with the model. Also used for referring back to previous conversations. This function can be also used when the user asks to do something that does not have a function yet'''

    Output Format - Single Function
    Output:
    {"assistant_reply":"insert lengthy assistant reply here","function_called":["function_name(parameter)"]}

    Output Format - Multiple Function
    Output:
    {"assistant_reply":"insert lengthy assistant reply here","function_called":["function_name_1(parameter)","function_name_2(parameter)"]}

    Example - Multi Turn Conversation. Follow this format for any function call.
    Input: hello there! Can you play me some music?
    Output:
    {"assistant_reply":"Hello! I'm Vivy, your personal AI assistant. I'd love to play some music for you. What genre or mood are you in the mood for?","function_called":["none()"]}

    Input: I am in the mood for some Pop
    Output:
    {"assistant_reply":"Great choice! Here are a few popular pop songs you might enjoy:
    Shape of You by Ed Sheeran
    Blinding Lights by The Weeknd
    Happier by Marshmello ft. Bastille
    Stitches by Shawn Mendes. Let me know if you would like me to play any of these songs!,"function_called":["none()"]}

    Input: Please play shape of you
    Output:
    {"assistant_reply":"Sure! You shall be able to hear that song right about now!","function_called":["play(Shape of you)"]}

    Below is the chat memory to help make your choices better:
    """
    prompt = message
    chat_memory+="user {}\n".format(prompt)
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
    #inputs

    
    llm_out = output['choices'][0]['text']
    chat_memory+="assistant {}\n".format(prompt)

    search_dict = eval(llm_out)
    search_list = search_dict["function_called"]

    
    opt = ""
    
    for i in search_list:     
        # Number Calculation
        if "calc_no" in i:
            result = re.search(r"'(\d+[\+\-\*\/]+\d+)'", str(i))
            if result:
                extracted_equation = result.group(1)
                opt += "The value of the function call " + str(i) + " is " + eval(extracted_equation)
                opt += "\n"

        # Internet Search
        elif "search" in i:
            link = ""
            mainp = ""
            
            pattern = r"\(([^)]+)\)"
            matches = re.findall(pattern, str(i))
            
            with DDGS() as ddgs:
                for r in ddgs.text(str(matches[0]), region='in-en', safesearch='off', timelimit='y',max_results=1):
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
                opt += "The value of function call " + str(i)+ " is " + mainp
                opt += "\n"
            
            else:
                opt += "The value of function call " + str(i)+ " is " + mainp
                opt += "\n"     

    if len(chat_memory) > 6000:
        chat_memory = chat_memory[:6000]
    
    return search_dict["assistant_reply"] + "\n" + opt

def realtime():
    import pyperclip as pc
    import pyautogui

    while True:
        compt = """You are an AI model who can read user's text. You have to help them write their messages, expand upon it, explain it or summarize it according to what the user wants. The user will mention what they want to do with the text before giving the input. If the user does not give the usage, make your guess depending on the content of the text, such as a email reply or content explanation. Commands are prefaced with a #.
        Example:
        #Rewrite this text
        #Reply to this email
        """
        prompt = pc.waitForNewPaste()
        output = llm(
        "<|im_start|>system {}<|im_end|>\n<|im_start|>user {}<|im_end|>\n<|im_start|>assistant".format(compt,prompt),
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
        llm_out = output['choices'][0]['text']
        pc.copy(llm_out)
        
        with pyautogui.hold('ctrl'):
            pyautogui.press(['v'])
        
        llm.reset()
        yield pc.paste()


# Main Code
c1 = gr.ChatInterface(chat,
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Enter Question", container=False, scale=7),
    title="AI Assistant",
    theme="soft",
    examples=["Good Morning!", "Google en passant", "what is 899*99/21"],
    clear_btn="Clear",)

with gr.Blocks() as c2:
    output = gr.Textbox(label="Output Box",)
    theme="soft"
    start = gr.Button("Start")
    start.click(fn=realtime, inputs=None, outputs=output, api_name="realtime")

demo = gr.TabbedInterface([c1, c2], ["Assistant Mode", "Realtime Mode"])
demo.launch()

# Fix search function, as its not working for some pages
