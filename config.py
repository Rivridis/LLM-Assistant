import json,re
print("Welcome to the configuration program!")
path=str(input("Please input your LLM-Assistant path: ")).replace("\\","/")
if(path[-1]=='/'):
    path=path[:-1]
conf=json.loads(open(f"{path}/config.json").read())
keys=conf.keys();
if(input("Do you wish to see the advanced configuration or just the basic configuration?(Type 1 or 2): ")=="1"):
    print(f"This is your actual configuration: {conf}")
    inpt=input("If you wish to modify it, type the name of the setting you want to change, else, type 'close'")
    if(inpt=="close" or (inpt in keys)==False):
        exit()
    val=input(f"type the new value for {inpt}")
    if((isInstance(val,int))==True):
        conf[inpt]=int(val)
        with open(f"{path}/config.json","w") as f:
            f.write(json.dumps(conf))
        input("The program will now exit, press enter")
        exit()
    conf[inpt]=val
    with open(f"{path}/config.json","w") as f:
        f.write(json.dumps(conf))
    input("The program will now exit, press enter")
    exit()
inp=input(f"This is your configuration:\n1. Path:{conf['path']}\n2. GPU offload/Layers(Makes the model run faster):{conf['gpu_offload']}\n3. CPU Threads(More is not always better):{conf['cpu_threads']}\n4. System Prompt:{conf['system_prompt']}\nEnter the number of the setting you want to change or type 'close' to exit: ")
if(inp=="1"):
    conf["path"]=input("Enter the new path: ")
elif(inp=="2"):
    conf["gpu_offload"]=int(input("Enter the number of layers: "))
elif(inp=="3"):
    conf["cpu_threads"]=int(input("Enter the number of threads: "))
else:
    conf["system_prompt"]=input("Enter the new System Prompt: ")
with open(f"{path}/config.json","w") as f:
    f.write(json.dumps(conf))
input("The program will now exit, press enter")
exit()


