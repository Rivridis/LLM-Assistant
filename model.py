import re
from llama_cpp import Llama
import chromadb
from duckduckgo_search import DDGS
import features
from functions import *
from trafilatura import fetch_url, extract
from messages import message, message_main, response_format, response_google, message_google, message_summary


chat_memory = ""
# Load the LLaMA model
llm = Llama(model_path=r"model\neuralhermes-2.5-mistral-7b.Q5_K_M.gguf",n_ctx=4098,n_gpu_layers=20)
client = chromadb.Client()
collection = client.get_or_create_collection(name="functions")
functions = [
    {"id": "1", "name": "search", "description": search_function},
    {"id": "2", "name": "weather", "description": weather_function},
    {"id": "3", "name": "play", "description": play_functions},
]

# Add function definitions to ChromaDB
for function in functions:
    collection.add(
        ids=[function["id"]],
        metadatas=[{"name": function["name"]}],
        documents=[function["description"]],
    )

def process_chat(text):
    print(text)
    global chat_memory
    message_main[1]["content"] = ""
    inp = text
    results = collection.query(
        query_texts=[inp],
        n_results=1  # Get the best match
        )

    message_main[0]["content"] =  message_main[0]["content"].split("MAIN")[0]
    message_main[0]["content"] += "\nMAIN" + "\n"
    message_main[0]["content"] += str(chat_memory) + "\n"
    
    message[1]["content"] = inp
    message[0]["content"] = message[0]["content"].split("MAIN")[0]
    message[0]["content"] += "\nMAIN" + "\n"
    message[0]["content"] += str(results["documents"][0][0])
    message[0]["content"] += chat_memory



    # Generate a response
    response = llm.create_chat_completion(
        messages= message,
        response_format= response_format,
        temperature=0.7,
    )

    func = eval(response["choices"][0]["message"]["content"])
    print(func)
    # Extract and print the JSON response

    function_result = ""
    if func.get("function_called") == "search":
        query = func.get("function_value")
        query_output = features.internet_search(query)
        message_google[1]["content"] = str(query_output)
        response = llm.create_chat_completion(
        messages= message_google,
        temperature=0.7,
        response_format= response_google,
        )

        value = response["choices"][0]["message"]["content"]
        print(value)
        selected_url = query_output[int(value)]
        url = selected_url["url"]

        website_data = features.read_website(url)
        print(website_data)
        website_data = website_data[:4000]

        message_summary[1]["content"] = str(inp) + "\n" + str(website_data)

        response = llm.create_chat_completion(
        messages= message_summary,
        temperature=0.7,
        )

        summary = response["choices"][0]["message"]["content"]
        chat_memory += " User Message:" + inp + "\n"
        chat_memory += " Assistant Response:" + summary + "\n"
        if len(chat_memory) > 5000:
            chat_memory = chat_memory[-5000:]
        
        return(summary)

    if  func.get("function_called") == "music":
        import pywhatkit
        match = func.get("function_value")
        print(match)
        pywhatkit.playonyt(str(match))
        function_result += f"Function Result: Song has been changed to {match}, which is playing now.\n"
    
    if  func.get("function_called") == "youtube":
        import pywhatkit
        match = func.get("function_value")
        print(match)
        pywhatkit.playonyt(str(match))
        function_result += f"Function Result: Youtube search video result is {match}, which is playing now.\n"

    if func.get("function_called") == "weather":
        location = func.get("function_value")
        if location == "":
            location = "Tokyo, Japan"
        weather_data = features.weather(location)
        print(weather_data)
        function_result += f"Interpret this weather data {str(weather_data)} in a user readable format\n"


    if func.get("function_called") == "none":
        function_result += " No function called, please answer the user's question"


    message_main[0]["content"] += function_result + "\n"
    message_main[1]["content"] += inp + "\n"
    print(message_main[1]["content"])

    if len(chat_memory) > 5000:
        chat_memory = chat_memory[-5000:]


    response = llm.create_chat_completion(
    messages= message_main,
    temperature=0.7,
    )

    out = response["choices"][0]["message"]["content"]

    chat_memory += " User Message:" + inp + "\n"
    chat_memory += " Assistant Response:" + out + "\n"


    if len(chat_memory) > 5000:
            chat_memory = chat_memory[-5000:]
    print(chat_memory)
    return(out)
