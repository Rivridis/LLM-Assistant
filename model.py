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
            "content": """You are a helpful function calling AI that outputs in JSON format. Do not reply to the user's questions. Details about how to use functions are given below. Strictly follow that. Your previous response is given below. Use that to call the correct function.
            
            Functions Available: search, weather, play, pause, read_mail, none, multi_turn_example.
            Function Descriptions:
            None Function
            Used when user is just chatting with the assistant, or the asisistant needs more information from the user
            {
                "function_called": "none",     
                "function_value": ""
            }
            """ ,
        },
        {"role": "user", "content": ""},
    ]

message_main = [
        {
            "role": "system",
            "content": """You are an AI assistant named Vivy, who responds to the the user's questions, using the value provided by the function call. Always follow the values provided in the function result below, and don't make up your own values. If there is any mistake in the provided function result and the user question, let the user know the call failed.
            You are provided with the chat memory of the conversation. Use it to answer the user's questions or help the user by asking for more information.
            EXAMPLE
            System: The value of function call is - weather is [40 celsius, 1013 hPa, Tokyo, Japan]
            User: Can you tell me the weather right now?
            
            Assistant: using the provided function result, The weather right now is 40 celsius with a pressure of 1013 hPa in Tokyo, Japan.
            """,
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
            "required": ["function_called"],
        },
    }

class MyCustomLLM():
    # Your ask function will always receive a list of prompts
    # The prompts are in open ai prompt format
    #  example: {"role": "system", "content": "You are a helpful assistant."}
    # If your model supports json format, use the format parameter to specify that to your model.
    def ask(self, prompts:list, format:str="", temperature:float=0.8):
        """
        Args:
            prompts (list): A list of prompts to ask.
            format (str, optional): The format of the response. Use "json" for json. Defaults to "".
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.
        """
        response = llm.create_chat_completion(
            messages= message_main,
            temperature=0.7,
        )
        return "Your llms response to the prompts goes here!" 
    
def process_chat(text):
    print(text)
    global chat_memory
    message_main[1]["content"] = ""
    inp = text
    results = collection.query(
        query_texts=[inp],
        n_results=1  # Get the best match
        )

    message_main[0]["content"] += str(chat_memory) + "\n"
    message[1]["content"] = inp
    message[0]["content"] += str(results["documents"][0][0])
    #print(results["documents"][0][0])
    message[0]["content"] += chat_memory
    #print(message[0]["content"])


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
            function_result += "The value of function call - search is " + mainp
            opt += "\n"
        
        else:
            function_result += "The value of function call - search is " + mainp
            function_result += "\n" 
            
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
        print(match)
        pywhatkit.playonyt(str(match))
        function_result += f"Function Result: Song has been changed to {match}, which is playing now.\n"


    if func.get("function_called") == "none":
        function_result += " No function called"


    message_main[0]["content"] += function_result + "\n"
    message_main[1]["content"] += inp + "\n"
    print(message_main[1]["content"])

    response = llm.create_chat_completion(
    messages= message_main,
    temperature=0.7,
    )

    out = response["choices"][0]["message"]["content"]

    chat_memory += " User Message:" + inp + "\n"
    chat_memory += " Function Called:" + str(func.get("function_called")) + "\n"
    chat_memory += " Assistant Response:" + function_result


    if len(chat_memory) > 2000:
            chat_memory = chat_memory[-2000:]
    return(out)
