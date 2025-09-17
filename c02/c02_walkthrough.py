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
