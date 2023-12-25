from llama_cpp import Llama
llm = Llama(model_path="B:\AI\Mistral\TheBloke\dolphin-2.2.1-mistral-7b.Q5_K_M.gguf\dolphin-2.2.1-mistral-7b.Q5_K_M.gguf", n_gpu_layers=30, n_ctx=3584, n_batch=521, verbose=True)
# adjust n_gpu_layers as per your GPU and model
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
print(output)
output = llm("Q: how many stars are there in the sky? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
print(output)