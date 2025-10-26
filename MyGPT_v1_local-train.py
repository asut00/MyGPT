import torch
import tiktoken
from src.GPTModel import GPTModel, generate_text_simple, generate
from src.utils import text_to_token_ids, token_ids_to_text
from src.Dataloader import create_dataloader_v1
from src.utils_train import train_model_simple


GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }

###################################
####### Untrained GPT Model #######
###################################

print()
print("===========================================")
print("=========== UNTRAINED GPT MODEL ===========")
print("===========================================")
print()

# Initiate model
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference

print("Untrained model created with the following configuration:")
for key, value in GPT_CONFIG_124M.items():
	print(f"- {key}: {value}")

# Run an inference with untrained model
print("-----")
print("Let's run an inference with our raw untrained model:")
start_context = "Every effort moves you"
print(f"Input text:\n>>> {start_context}")
tokenizer = tiktoken.get_encoding("gpt2")

tokens = text_to_token_ids(start_context,tokenizer)

token_ids = generate_text_simple(
	model=model,
	idx=text_to_token_ids(start_context,tokenizer),
	max_new_tokens=10,
	context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n>>> ", token_ids_to_text(token_ids, tokenizer))

print("-----")
print("As we can see, our untrained model does not produce a good text because... it has not been trained yet!")

######################################
####### Training our GPT Model #######
######################################

print()
print("==============================================")
print("=========== TRAINING OUR GPT MODEL ===========")
print("==============================================")
print()

print("Let's train our model and try again...")

file_path = "the-verdict.txt"

with open(file_path, "r", encoding="utf-8") as file:
	text_data = file.read()

print(f"Loaded trainging data from: {file_path}")


# Splitting into train and validation set
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print(f"Created training and validation data loaders")

# Setting up device
if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps") # allows Apple Silicon chip usage
else:
   device = torch.device("cpu")

model.to(device)

print(f"Using {device} device.")

torch.manual_seed(123)

# Train the model
print("="*50)
print("Starting Training...")

import time
start_time = time.time()

torch.manual_seed(123) 

model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)


num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Compute execution time
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print("-----")
print(f"Training completed in {execution_time_minutes:.2f} minutes.")