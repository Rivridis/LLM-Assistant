# LLM-Assistant
LLM-Assistant is a browser interface based on Gradio that interfaces with local LLMs to call functions and act as a general assistant.

## Features
* Works with any instruct-finetuned LLM
* Can search for information (RAG)
* Knows when to call functions
* Added Realtime mode for working across the system

## Upcoming Features
* Vector Database
* Function execution support
* Error handling
* GBNF Grammar output structure

## Current Bugs
* Search feature might crash at times
* No error handling, causing the code to crash when LLM produces unparsable output
* No stop button for Realtime mode

## Setup
### Setup on Windows 10/11
1. Clone repo to a virtual environment
2. Install requirements.txt
3. Download and place LLM model in model folder
4. Run main.py

### Usage
* Use Assistant mode for general chat, and calling functions to execute like playing music.
* Use Realtime mode for editing a word document or replying to an email in realtime, directly by copying a selection and waiting for the output.
The output gets auto pasted at cursor location.

## Images
![image](https://github.com/Rivridis/LLM-Assistant/assets/97879757/5f7b5ada-119a-4d5f-9eff-75a4360ab3cc)

![image](https://github.com/Rivridis/LLM-Assistant/assets/97879757/2897b287-95b7-4a24-9979-1abe2325013d)



