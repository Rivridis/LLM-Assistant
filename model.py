from llama_cpp import Llama
import chromadb
from duckduckgo_search import DDGS
from functions import *
from trafilatura import fetch_url, extract


chat_memory = ""
# Load the LLaMA model
llm = Llama(model_path=r"model\neuralhermes-2.5-mistral-7b.Q5_K_M.gguf", chat_format="chatml",n_ctx=4098,n_gpu_layers=20)
client = chromadb.Client()
collection = client.get_or_create_collection(name="functions")
functions = [
    {"id": "1", "name": "search", "description": search_function},
    {"id": "2", "name": "weather", "description": weather_function},
    {"id": "3", "name": "play", "description": play_functions},
    {"id": "4", "name": "pause", "description": pause_function},
    {"id": "5", "name": "read_mail", "description": read_mail_function},
    {"id": "6", "name": "none", "description": none_function},
    {"id": "7", "name": "multi_turn_example", "description": multi_turn_example},
]

# Add function definitions to ChromaDB
for function in functions:
    collection.add(
        ids=[function["id"]],
        metadatas=[{"name": function["name"]}],
        documents=[function["description"]],
    )
message = [
        {
            "role": "system",
            "content": "You are a helpful function calling AI that outputs in JSON format. Always respond in one word and follow the format of giving function called, and function value is the parameter that is to be called. Do not reply to the user's questions. Details about how to use functions are given below. Strictly follow that. Chat memory is given below as well. Use that to call the correct function value" ,
        },
        {"role": "user", "content": ""},
    ]

message_main = [
        {
            "role": "system",
            "content": " You are an AI Assistant named Vivy, who responds to the user with helpful information, tips, and jokes just like Jarvis from the marvel universe. You must be answer all the questions truthfully. You will be given the function call value that you called earlier. Use the function call value to formulate your answer. If the function call value is none, then you can chat with the user. You can also refer to the previous conversation. You can also ask the user for more information if needed. Chat memory will be provided below. Use the user input to figure out if the function was called correctly. Don't respond to the user's function call without a valid function value. List of functions available are search, weather, play, pause, read_mail, none, multi_turn_example",
        },
        {"role": "user", "content": ""},
    ]


response_format = {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "function_called": {
                    "type": "string",
                    "enum": ["play", "weather", "none", "search"]
                },
                "function_value": {"type": "string"}
            },
            "required": ["function_called", "function_value"],
        },
    }

def process(text):
    inp = text
    results = collection.query(
        query_texts=[inp],
        n_results=1  # Get the best match
        )

    message[1]["content"] = inp
    message[0]["content"] += str(results["documents"][0][0])
    message[0]["content"] += chat_memory
    #print(message[0]["content"])

    # Generate a response
    response = llm.create_chat_completion(
        messages= message,
        response_format= response_format,
        temperature=0.7,
    )

    func = eval(response["choices"][0]["message"]["content"])
    # Extract and print the JSON response

    opt = ""
    print(func.get("function_called"))
    print(func.get("function_value"))
    if func.get("function_called") == "search":

        link = []
        mainp=""
        match = func.get("function_value")
        
        results = DDGS().text(match, region='wt-wt', safesearch='off', timelimit='d', max_results=2)
        for i in results:
            link.append(i["href"])

        content = ""
        for i in link:
            downloaded = fetch_url(i)
            result = extract(downloaded)
            content += str(result)
            content += "Next Search Result\n"

        mainp += content
        print(mainp)      
        
        if len(mainp) > 1500:
            mainp= mainp[:1500]
            opt += "The value of function call - search is " + mainp
            opt += "\n"
        
        else:
            opt += "The value of function call - search is " + mainp
            opt += "\n" 
            
    if  func.get("function_called") == "youtube":
        match = func.get("function_value")
        results = DDGS().videos(
        keywords=str(match),
        region="wt-wt",
        safesearch="off",
        timelimit="w",
        resolution="high",
        duration="medium",
        max_results=5,
        )
        val = ""
        for i in results:
            val += f"{str(i['content'])}\n{str(i['description'])}\n"
        opt += f"The value of function call - youtube is {val}\n"

    if  func.get("function_called") == "play":
        import pywhatkit
        match = func.get("function_value")
        pywhatkit.playonyt(str(match))
        opt += f"Function call - play is successful. Current Song Playing: {str(match)}\n"

    if func.get("function_called") == "none":
        opt += "No function called"



    message_main[0]["content"] += str(chat_memory)
    message_main[0]["content"] += opt
    message_main[1]["content"] += inp + "\n"

    print(len(message_main[0]["content"]))
    response = llm.create_chat_completion(
    messages= message_main,
    temperature=0.7,
    )

    out = response["choices"][0]["message"]["content"]
    print(out)

    chat_memory += "User Message:" + inp + "\n"
    chat_memory += "Function Called:" + str(func.get("function_called")) + "\n"
    chat_memory += "Assistant Response:" + out + "\n"

    print(len(chat_memory))

    if len(chat_memory) > 2000:
            chat_memory = chat_memory[-2000:]

