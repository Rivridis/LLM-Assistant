# LLM-Assistant
LLM-Assistant is a browser interface based on Gradio that interfaces with local LLMs to call functions and act as a general assistant.

## Features
* Works with any instruct-finetuned LLM
* Can search for information (RAG)
* Knows when to call functions
* Realtime mode for working across the system
* Answers question from PDF files
* GPU support(Flash attention supported too!)
  
## Roadmap
* Voice access
* More functions
* Add support for custom System prompt
  
## Current Bugs
* Rare crashes

## Changelog
* Fixed search feature
* Youtube video search
* File Upload
* Added GPU support
* Added Flash Attention Support
+ Additional support has been added for easily changing settings
+ Added chat history support
+ Weather function working
+ Current date & time

## Setup
### Setup on Windows 10/11
1. Clone repo to a virtual environment
2. If you have an Nvidia GPU and have the CUDA drivers installed run 'pip install llama_cpp_python==0.2.77 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124', if not, run 'pip install llama_cpp_python==0.2.77'
3. Install requirements.txt
4. Download and place LLM model in model folder
5. Run config.py
6. Run main.py

### Setup on Linux
1. Clone repo to a virtual environment (python -m venv "virtualenviromentdirectory")
2. If you have an Nvidia GPU and have the CUDA drivers installed run 'pip install llama_cpp_python==0.2.77 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124', if not, run 'pip install llama_cpp_python==0.2.77'
3. Install requirements.txt
4. Download and place LLM model in model folder
5. Run config.py
6. Run main.py

### Usage
* Use Assistant mode for general chat, and calling functions to execute like playing music, as well as PDF question answering
* Use Realtime mode for editing a word document or replying to an email in realtime, directly by copying a selection and waiting for the output.
The output gets auto pasted at cursor location.

## Images

![image](https://github.com/Rivridis/LLM-Assistant/assets/97879757/eb5f1d46-a607-40b1-8275-19c92fafa14f)
![image](https://github.com/Rivridis/LLM-Assistant/assets/97879757/2897b287-95b7-4a24-9979-1abe2325013d)



