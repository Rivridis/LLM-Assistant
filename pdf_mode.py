from json import load
import chromadb
from model import llm
from pathlib import Path
from pypdf import PdfReader
from urllib.parse import urlparse, unquote

class PDFChatAssistant:
    def __init__(self):
        self.chat_memory = ""
        self.filepath = ""
        self.client_pdf = chromadb.Client()
        self.collection = None
        self.message = [
            {
                "role": "system",
                "content": """
                You are an AI Assistant who is trained for answering user questions from the given PDF. You are provided with the relevant snippet from the PDF. Only use the given snippet to answer the user's question. If no snippets are given, or the user's question is not there in the given context, let the user know that their question can't be answered. Parse the given context, so it can be readable by the user, and explain the user's query using the context. Clearly mention the page numbers (ids) from which the context is taken.

                EXAMPLE
                Context:
                Snippet from PDF related to query
                Page Numbers:
                []

                MAIN
                """
            },
            {"role": "user", "content": ""},
        ]

    
    def parse_file_url(self, url):
        parsed = urlparse(url)
        # Remove leading slash on Windows paths
        path = unquote(parsed.path)
        if path.startswith('/') and ':' in path:
            path = path[1:]
        return path

    def load_pdf(self, file_path):
        file_name = Path(file_path).stem
        fl = file_name.lower().replace(" ", "_")
        pglst = []
        reader = PdfReader(str(file_path))
        number_of_pages = len(reader.pages)
        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            pglst.append(text)
        self.collection = self.client_pdf.get_or_create_collection(name=str(fl))
        self.collection.add(
            documents=pglst,
            ids=[f"{i}" for i in range(len(pglst))]
        )

    def process_chat(self, text, url):
        if llm == None:
            return("Model not loaded. Please place your model in the 'model' directory and restart the app.")
        if self.filepath == "":
            self.filepath = self.parse_file_url(url)
            self.load_pdf(self.filepath)
        results = self.collection.query(
            query_texts=[text],
            n_results=2,
        )
        print(results)

        self.message[1]["content"] = text
        self.message[0]["content"] += "Context:" + "\n" + str(results["documents"]) + "\nPage Numbers:\n" + str(results["ids"])

        main_instructions = self.message[0]["content"].split("MAIN")[0]
        chat_history = self.message[0]["content"][len(main_instructions):]
        max_length = 5000
        if len(self.message[0]["content"]) > max_length:
            # Truncate chat_history to keep the most recent part, but keep main_instructions intact
            truncated_history = chat_history[-(max_length - len(main_instructions)):]
            self.message[0]["content"] = main_instructions + "MAIN" + truncated_history

        response = llm.create_chat_completion(
        messages= self.message,
        temperature=0.7,
        )
        
        self.message[0]["content"] += "Chat Memory:\n"
        self.message[0]["content"] += "User: " + text
        self.message[0]["content"] += "Assistant: " + str(response["choices"][0]["message"]["content"])

        return(str(response["choices"][0]["message"]["content"]))



