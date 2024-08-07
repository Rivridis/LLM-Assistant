import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

import json

print("Welcome to the configuration program!")

path = filedialog.askopenfilename()
config = {"path": str(path), "gpu_offload": 20, "context_length": 2048, "cpu_threads": 4, "n_batch": 512, "flash_attention": 1, "max_tokens": -1, "temperature": 0.8, "top_k": 40, "repeat_penalty": 1.1, "min_p_sampling": 0.05, "max_p_sampling": 0.05}

with open('config.json', 'w') as f:
    json.dump(config, f)

Op = input("Press 1 to enter advanced configuration mode or press 2 to exit configuration: ")
if Op == "1":
    print("Welcome to the advanced configuration mode!")
    print("Here are the settings you can change:")
    print("1. GPU offload - This setting determines how many layers of the model will be offloaded to the GPU. Increasing this value will make the model run faster.")
    print("2. CPU Threads - This setting determines how many threads the model will use. Increasing this value might make the model run faster but may cause instability.")

    while True:
        print(f"This is your current configuration: \n{config}")
        choice = input("Enter the number of the setting you want to change or type 'exit' to exit: ")
        if choice == "1":
            val = int(input("Enter the new value for GPU offload: "))
            config["gpu_offload"] = val
            with open('config.json', 'w') as f:
                f.write(json.dumps(config))
        elif choice == "2":
            val = int(input("Enter the new value for CPU Threads: "))
            config["cpu_threads"] = val
            with open('config.json', 'w') as f:
                f.write(json.dumps(config))
        else:
            print("Exiting configuration")
            break
    
else:
    print("Exiting configuration")


