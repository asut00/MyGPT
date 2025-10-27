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
####### Temperature scaling #######
###################################

# - To add variety, we can sample the next token using The `torch.multinomial(probs, num_samples=1)`, sampling from a probability distribution
# - Here, each index's chance of being picked corresponds to its probability in the input tensor

import torch
import matplotlib.pyplot as plt

vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
} 

inverse_vocab = {v: k for k, v in vocab.items()}

print("vocab:")
print(vocab)
print("inverse_vocab:")
print(inverse_vocab)

# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)
print("probas:")
print(probas)
next_token_id = torch.argmax(probas).item()
print("next_token_id:")
print(next_token_id)

# The next generated token is then as follows:
print(inverse_vocab[next_token_id])

print("-"*50)

def print_sampled_tokens(probas):
    torch.manual_seed(123) # Manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)


# - We can control the distribution and selection process via a concept called temperature scaling
# - "Temperature scaling" is just a fancy word for dividing the logits by a number greater than 0
# - Temperatures greater than 1 will result in more uniformly distributed token probabilities after applying the softmax
# - Temperatures smaller than 1 will result in more confident (sharper or more peaky) distributions after applying the softmax

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# Temperature values
temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence

# Calculate scaled probabilities
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]

print("next_token_logits:")
print(next_token_logits)
print("scaled_probas:")
print(scaled_probas)

print('='*50)

# Plotting
x = torch.arange(len(vocab))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()

plt.tight_layout()
plt.savefig("temperature-plot.pdf")
# plt.show()

# - We can see that the rescaling via temperature 0.1 results in a sharper distribution, approaching `torch.argmax`, such that the most likely word is almost always selected:

print("Repartition with 0.1 temperature:")
print_sampled_tokens(scaled_probas[1])

print('-----')
print("Repartition with 5 temperature:")
print_sampled_tokens(scaled_probas[2])


##############################
####### Top-k sampling #######
##############################

# - To be able to use higher temperatures to increase output diversity and to reduce the probability of nonsensical sentences, we can restrict the sampled tokens to the top-k most likely tokens:

top_k = 3

top_logits, top_pos = torch.topk(next_token_logits, top_k)

print("="*50)
print("next_token_logits:")
print(next_token_logits)
print("Top logits:", top_logits)
print("Top positions:", top_pos)

print("="*50)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")),
    other=next_token_logits
)

print("new_logits:")
print(new_logits)

topk_probas = torch.softmax(new_logits, dim=0)
print("topk_probas:")
print(topk_probas)


######################################################
####### Modifying the text generation function #######
######################################################



# - The previous two subsections introduced temperature sampling and top-k sampling
# - Let's use these two concepts to modify the `generate_simple` function we used to generate text via the LLM earlier, creating a new `generate` function:


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            tempered_logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(tempered_logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

from GPTModel import GPTModel
import tiktoken

torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)
tokenizer = tiktoken.get_encoding("gpt2")


def text_to_token_ids(text, tokenizer):
	encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
	encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
	return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
	flat = token_ids.squeeze(0) # remove batch dimension
	return tokenizer.decode(flat.tolist())

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# Loading and saving model weights in PyTorch


###########################################################
####### Loading and saving model weights in PyTorch #######
###########################################################


# - The recommended way in PyTorch is to save the model weights, the so-called `state_dict` via by applying the `torch.save` function to the `.state_dict()` method:

# Saving Model:

torch.save(model.state_dict(), "model.pth")

print("Model saved !")

# Loading model:

model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()

print("Model loaded !")

# We can save the model with the optimizer:

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) # had to add this line here to avoid error but normally the optimizer should have been trained with the rest of the model

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    "model_and_optimizer.pth"
)

# And load with optimizer :

checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();

######################################################
####### Loading pretrained weights from OpenAI #######
######################################################

from importlib.metadata import version

print("TensorFlow version:", version("tensorflow"))
print("tqdm version:", version("tqdm"))