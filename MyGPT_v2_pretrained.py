import torch
import tiktoken
import numpy as np
from src.GPTModel import GPTModel
from src.utils_train import generate
from src.utils import GPT_CONFIG_124M
from src.gpt_download import download_and_load_gpt2
from src.utils import text_to_token_ids, token_ids_to_text
from src.gpt_loadweights import load_weights_into_gpt

print()
print("============================================")
print("=========== PRETRAINED GPT MODEL ===========")
print("============================================")
print()

print("Downloading model weights for 124 million parameter model...")

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# Adapt Config
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

# Create model from our GPTModel class
gpt = GPTModel(NEW_CONFIG)
gpt.eval(); # Activate eval mode

# Assign the OpenAI weights to the corresponding weight tensors in our `GPTModel` instance
load_weights_into_gpt(gpt, params)

# Setting up device
if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps") # allows Apple Silicon chip usage
else:
   device = torch.device("cpu")

print(f"Using {device} device.")
gpt.to(device);

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

system_prompt = "You are a helpful, honest, and harmless AI assistant. Always provide accurate, concise, and clear answers to the user's questions. If you don’t know the answer, you should say so. "

# Chat loop
while True:
    user_input = input("MyGPT: Hello, how can I help you ? (enter 'exit' to end chat)\n> ")

    if user_input == "exit":
        exit()

    prompt = system_prompt + user_input

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=128,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )

    text_answer = token_ids_to_text(token_ids, tokenizer)

    text_answer = text_answer[len(prompt):] + "|end of context|" # Remove initial prompt from final answer

    text_answer = text_answer.lstrip() # Remove extraneous spaces in the beginning of answer string

    print("MyGPT:", text_answer)