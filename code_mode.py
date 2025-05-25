from model import llm

message = [
        {
            "role": "system",
            "content": """You are given a code snippet by the user, as well as the user query. Use the user query to modify the code snippet to make it work, such as generating code, big fixes or adding new features. Always respond in the correct code syntax and format, and do not add any extra text or response. If the user query is not related to generating or modifying coding, let the user know that you cannot help with that. If the user wants to generate code, create the code in the correct programming language and format it correctly. If the user wants to fix a bug, identify the bug and provide a corrected version of the code. If the user wants to add a new feature, implement that feature in the code snippet.
            Your default programming language is Python. Respond only with the code.
            """ ,
        },
        {"role": "user", "content": ""},
    ]

def process_chat(text, code):
    if llm == None:
     return("Model not loaded. Please place your model in the 'model' directory and restart the app.")
    message[1]["content"] = ""

    message[1]["content"] = text + "\n"
    message[1]["content"] += code  + "\n"


    response = llm.create_chat_completion(
        messages= message,
        temperature=0.7,
    )

    return(str(response["choices"][0]["message"]["content"]))
