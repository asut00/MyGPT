from importlib.metadata import version

print(f'torch version: {version('torch')}')
print(f'tiktoken version: {version('tiktoken')}')

# Import text source
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

#################### Developing tokenizer ####################

# We use the regular expression library to split on whitespaces
import re

text = "Hello, world. Is this-- a test?"

# This regular expression splits on whitespaces and signs (but keeps them in the generated list as tokens themselves)
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
print(result)

# Remove empty strings
stripped_result = [item.strip() for item in result if item.strip()]
# This should also work but oh well
# basic_result = [item for item in result if item]

print(f"stripped_result: {stripped_result}")


#################### Apply tokenizer to text ####################

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

print(len(preprocessed))

#################### Generate token IDs ####################

# Turn list into set in order to delete doubles
preprocessed_set = set(preprocessed)

# print(f'preprocessed_set: {preprocessed_set}')

sorted_set = sorted(preprocessed_set)

# print(f'sorted_set: {sorted_set}')

token_dict = {token:integer for integer,token in enumerate(sorted_set)}

print(f'type(vocab): {type(token_dict)}')

print('token_dict:')
# print(token_dict)

#################### Create a Tokenizer class ####################

class SimpleTokenizerV1:
    def __init__(self, token_dict):
        # {Token : token-id} dict 
        self.str_to_int = token_dict
        # {token-id : Token} dict
        self.int_to_str = {i:s for s,i in token_dict.items()}
    
    def encode(self, text):
        # Split input text into tokens
        # This regular expression splits on whitespaces and signs (but keeps them in the generated list as tokens)
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Find the associated token-ids in the Tokenizer vocab dict
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    def decode(self, ids):
        # Find the tokens associated to token-ids
        text = ' '.join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

#################### Use our token dict on our class ####################

tokenizer = SimpleTokenizerV1(token_dict)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print('-----')
print("ids:")
print(ids)

decoded_ids = tokenizer.decode(ids)
print('-----')
print("decoded_ids:")
print(decoded_ids)

#################### Adding special context ####################

text = 'Hello, do you like tea. Is this-- a test?'

# tokenizer.encode(text)
# >>> generates an error because Hello is not in the token-dict / vocab

# Extend the dictionary
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(['<|endoftext|>', '<|unk|>'])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print('-----')
print(f'len(vocab): {len(vocab)}')
print('list(vocab.items())[-5:]')
print(list(vocab.items())[-5:])

# Little refresher on dict methods
# print('list(vocab)[-5:]')
# print(list(vocab)[-5:])
# print('list(vocab.keys())[-5:]')
# print(list(vocab.keys())[-5:])
# print('list(vocab.values())[-5:]')
# print(list(vocab.values())[-5:])

# Remake the SimpleTokenizerClass : 
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        # Split input text into tokens
        # This regular expression splits on whitespaces and signs (but keeps them in the generated list as tokens)
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Replace unidentified items (absent of the vocabulary) by the <|unk|> context token
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        # Find the associated token-ids in the Tokenizer vocab dict
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    def decode(self, ids):
        # Find the tokens associated to token-ids
        text = ' '.join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# Test the new tokenizer
tokenizer = SimpleTokenizerV2(vocab)

text1 = 'Hello, do you like tea?'
text2 = 'In the sunlit terraces of the palace.'

text = " <|endoftext|> ".join((text1,text2))

print('------')
print(text)

encoded = tokenizer.encode(text)

print('------')
print(f'encoded: {encoded}')

decoded = tokenizer.decode(tokenizer.encode(text))

print(f'decoded: {decoded}')

#################### BytePair encoding ####################


import importlib
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding('gpt2')

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

print('len(text.split()):')
print(len(text.split())) #

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print('-----')
print('integers:')
print(integers)
print(f'len(integers): {len(integers)}') # 

strings = tokenizer.decode(integers)

print('-----')
print('strings:')
print(strings)

print('-----')
print('decoding integers:')
for elem in integers:
    token = tokenizer.decode([elem])
    print(f'|{token}|', end=' ')
print()

# Prints :
# >>> |Hello| |,| | do| | you| | like| | tea| |?| | | |<|endoftext|>| | In| | the| | sun| |lit| | terr| |aces| |of| | some| |unknown| |Place| |.|

#################### Data sampling with a sliding window ####################

# We train LLMs to generate one word at a time, so we want to prepare the training 
# data accordingly where the next word in a sequence represents the target to predict: 

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

num_of_words = len(raw_text.split())
print('-----')
print(f'num_of_words: {num_of_words}')

enc_text = tokenizer.encode(raw_text)
print(f'len(enc_text): {len(enc_text)}')

enc_sample = enc_text[50:]
print(f'enc_sample: {enc_sample}')

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f'x: {x}')
print(f'y: {y}')

# Figurative loop of input and prediction for each sliding windows
for i in range(1, context_size+1):
    input = enc_sample[:i]
    prediction = enc_sample[i]
    print(f'{input} -> {prediction}')

#### Implement dataloader ####

import torch
print('PyTorch version:', torch.__version__)

# Create dataset and dataloader that extract chunks from the input text dataset

from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        assert len(token_ids) > length, 'Number of tokenized inputs must at least be equal to length+1'

        for i in range(0, len(token_ids) - length, stride):
            input_chunk = token_ids[i:i+length]
            target_chunk = token_ids[i+1:i+length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

# Let's test the dataloader with a batch size of 1 for an LLM with a context size of 4:

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

# dataloader returns an iterable, so in order to check we have to iterate on it
data_iter = iter(dataloader)
first_batch = next(data_iter)
print('----')
print(f'data_iter: {data_iter}')
print('first_batch:')
print(first_batch)
second_batch = next(data_iter)
print('second_batch:')
print(second_batch)
# Prints:
# first_batch:
# [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
# second_batch:
# [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]


# Or with loop iterating on dataloader
for i, elem in enumerate(dataloader):
    print(f'elem {i}:')
    print(elem)
    if i >= 1:
        break

# Verification
tokenizer = tiktoken.get_encoding('gpt2')
check = tokenizer.encode(raw_text)
print(f'check[:5]: {check[:5]}')
# Prints:
# check[:5]: [40, 367, 2885, 1464, 1807]

# Other config
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print('----')
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# Prints:
# Inputs:
#  tensor([[   40,   367,  2885,  1464],
#         [ 1807,  3619,   402,   271],
#         [10899,  2138,   257,  7026],
#         [15632,   438,  2016,   257],
#         [  922,  5891,  1576,   438],
#         [  568,   340,   373,   645],
#         [ 1049,  5975,   284,   502],
#         [  284,  3285,   326,    11]])

# Targets:
#  tensor([[  367,  2885,  1464,  1807],
#         [ 3619,   402,   271, 10899],
#         [ 2138,   257,  7026, 15632],
#         [  438,  2016,   257,   922],
#         [ 5891,  1576,   438,   568],
#         [  340,   373,   645,  1049],
#         [ 5975,   284,   502,   284],
#         [ 3285,   326,    11,   287]])

#################### Creating token embeddings ####################

# Suppose we have the following four input examples with input ids 2, 3, 5, and 1 (after tokenization):

input_ids = torch.tensor([2, 3, 5, 1])

# For the sake of simplicity, suppose we have a small vocabulary of only 6 words and we want to create embeddings of size 3:

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print('-----')
print('embedding_layer.weight:')
print(embedding_layer.weight)
# Prints:
# tensor([[ 0.3374, -0.1778, -0.1690],
#         [ 0.9178,  1.5810,  1.3010],
#         [ 1.2753, -0.2010, -0.1606],
#         [-0.4015,  0.9666, -1.1481],
#         [-1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096]], requires_grad=True)
# Where each lines is the embedding for a token of the vocab

# To convert a token with id 3 into a 3-dimensional vector, we do the following:
print('-----')
print('embedding_layer(torch.tensor([3])))')
print(embedding_layer(torch.tensor([3]))) # we get the embedinng vector that is at index 3

# To embed all four `input_ids` values above, we do
print('-----')
print('embedding_layer(input_ids):')
print(embedding_layer(input_ids)) # The torch.nn.Embedding() takes a list of TokenIDs as argument and returns the embedding corrsponding to these ids (which are probably the vector at the indexes corresponding to the ID)

# >>> an embeding layer is essentially a look-up operation

#################################################################
#################### Encoding word positions ####################
#################################################################

# Embedding layer convert IDs into identical vector representations regardless of where they are located in the input sequence
# Positional embeddings are combined with the token embedding vector to form the input embeddings for a large language model

# The BytePair encoder has a vocabulary size of 50,257:
# Suppose we want to encode the input tokens into a 256-dimensional vector representation:

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# the token embedding layer is a list of vectors, each vector being an embedding of a token of the vocab

# If we sample data from the dataloader, we embed the tokens in each batch into a 256-dimensional vector
# If we have a batch size of 8 samples/input/phrase/text-extract with 4 tokens each, this results in a 8 x 4 x 256 tensor:

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print('-----')
print('Token IDs:\n', inputs)
print("\nInputs shape:\n", inputs.shape)

# Prints:
# Token IDs:
#  tensor([[   40,   367,  2885,  1464],
#         [ 1807,  3619,   402,   271],
#         [10899,  2138,   257,  7026],
#         [15632,   438,  2016,   257],
#         [  922,  5891,  1576,   438],
#         [  568,   340,   373,   645],
#         [ 1049,  5975,   284,   502],
#         [  284,  3285,   326,    11]])
# Inputs shape:
#  torch.Size([8, 4])

token_embeddings = token_embedding_layer(inputs)
print('-----')
print('token_embeddings:')
print(token_embeddings)
print(f'token_embeddings.shape: {token_embeddings.shape}')

# Prints :
# token_embeddings:
# tensor([[[ 0.4913,  1.1239,  1.4588,  ..., -0.3995, -1.8735, -0.1445],
#          [ 0.4481,  0.2536, -0.2655,  ...,  0.4997, -1.1991, -1.1844],
#          [-0.2507, -0.0546,  0.6687,  ...,  0.9618,  2.3737, -0.0528],
#          [ 0.9457,  0.8657,  1.6191,  ..., -0.4544, -0.7460,  0.3483]],

#         [[ 1.5460,  1.7368, -0.7848,  ..., -0.1004,  0.8584, -0.3421],
#          [-1.8622, -0.1914, -0.3812,  ...,  1.1220, -0.3496,  0.6091],
#          [ 1.9847, -0.6483, -0.1415,  ..., -0.3841, -0.9355,  1.4478],
#          [ 0.9647,  1.2974, -1.6207,  ...,  1.1463,  1.5797,  0.3969]],

#         [[-0.7713,  0.6572,  0.1663,  ..., -0.8044,  0.0542,  0.7426],
#          [ 0.8046,  0.5047,  1.2922,  ...,  1.4648,  0.4097,  0.3205],
#          [ 0.0795, -1.7636,  0.5750,  ...,  2.1823,  1.8231, -0.3635],
#          [ 0.4267, -0.0647,  0.5686,  ..., -0.5209,  1.3065,  0.8473]],

#         ...,

#         [[-1.6156,  0.9610, -2.6437,  ..., -0.9645,  1.0888,  1.6383],
#          [-0.3985, -0.9235, -1.3163,  ..., -1.1582, -1.1314,  0.9747],
#          [ 0.6089,  0.5329,  0.1980,  ..., -0.6333, -1.1023,  1.6292],
#          [ 0.3677, -0.1701, -1.3787,  ...,  0.7048,  0.5028, -0.0573]],

#         [[-0.1279,  0.6154,  1.7173,  ...,  0.3789, -0.4752,  1.5258],
#          [ 0.4861, -1.7105,  0.4416,  ...,  0.1475, -1.8394,  1.8755],
#          [-0.9573,  0.7007,  1.3579,  ...,  1.9378, -1.9052, -1.1816],
#          [ 0.2002, -0.7605, -1.5170,  ..., -0.0305, -0.3656, -0.1398]],

#         [[-0.9573,  0.7007,  1.3579,  ...,  1.9378, -1.9052, -1.1816],
#          [-0.0632, -0.6548, -1.0296,  ..., -0.9538, -0.5026, -0.1128],
#          [ 0.6032,  0.8983,  2.0722,  ...,  1.5242,  0.2030, -0.3002],
#          [ 1.1274, -0.1082, -0.2195,  ...,  0.5059, -1.8138, -0.0700]]],
#        grad_fn=<EmbeddingBackward0>)
# token_embeddings.shape: torch.Size([8, 4, 256])

# >>> each token ID in each sample has been replaced with the corresponding embedding vector of size 256
# >>> so we have a batch of 8 samples/input/phrase/text-extract, each one containing 4 token, each token being represented by an embedding vector of size 256

# Positional embedding :
# GPT-2 uses absolute position embeddings, so we just create another embedding layer

context_length = max_length
# print(f'context_length: {context_length}')
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# >>> En gros, the first arg of torch.nn.Embedding() gives the number of configurations that we want to be able to describe with -> i.e. the number of embeddings.

print('-----')
print('pos_embedding_layer.weight:')
print(pos_embedding_layer.weight)
# print('pos_embedding_layer.shape:')
# print(pos_embedding_layer.shape)

pos_embeddings = pos_embedding_layer(torch.arange(max_length)) # torch.arange(max_length): tensor([0, 1, 2, 3])
# Reminder : The torch.nn.Embedding() takes a list of TokenIDs as argument and returns the embedding corrsponding to these ids (which are probably the vector at the indexes corresponding to the ID)
# In this case the list it takes in argument is (should be ?) a list of the position of the token in the sample, and returns an embedding describing this position


# To create the input embeddings used in an LLM, we simply add the token and the positional embeddings:
input_embeddings = token_embeddings + pos_embeddings

print('input_embeddings:')
print(input_embeddings)


print(f'input_embeddings.shape: {input_embeddings.shape}')

# Summary :
# - In the initial phase of the input processing workflow, the input text is segmented into separate tokens
# - Following this segmentation, these tokens are transformed into token IDs based on a predefined vocabulary
# - Then these TokenIDs are converted to the corresponding embeddings
# - Then these TokenID embeddings are merged with the positionnal embeddings. Through a simple addition : we add the positional embedding (of size 256) to each tokenID embedding.
# >>> This gives us our final input embeddings