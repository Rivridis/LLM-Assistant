from llama_cpp import Llama
import chromadb
from functions import *

chat_memory = ""
# Load the LLaMA model
llm = Llama(model_path=r"model\neuralhermes-2.5-mistral-7b.Q5_K_M.gguf", chat_format="chatml")
client = chromadb.Client()
collection = client.get_or_create_collection(name="functions")
functions = [
    {"id": "1", "name": "search", "description": search_function},
    {"id": "2", "name": "weather", "description": weather_function},
    {"id": "3", "name": "play", "description": play_function},
    {"id": "4", "name": "pause", "description": pause_function},
    {"id": "5", "name": "read_mail", "description": read_mail_function},
    {"id": "6", "name": "youtube", "description": youtube_function},
    {"id": "7", "name": "none", "description": none_function},
    {"id": "8", "name": "multi_turn_example", "description": multi_turn_example},
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
            "content": "You are a helpful function calling AI that outputs in JSON format. Always respond in one word and follow the format of giving function called, and function value is the parameter that is to be called. Do not reply to the user's questions. Always use the none function with empty as parameter for most user queries. Details about how to use functions are given below. Strictly follow that." ,
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


inp = input("Enter a message: ")

results = collection.query(
    query_texts=[inp],
    n_results=1  # Get the best match
    )

print(results)
message[1]["content"] = inp
message[0]["content"] + str(results["documents"])

# Generate a response
response = llm.create_chat_completion(
    messages= message,
    response_format= response_format,
    temperature=0.7,
)

# Extract and print the JSON response
print(response["choices"][0]["message"]["content"])
