from importlib.metadata import version

print("torch version:", version("torch"))

import torch

############################################################
################### Simplified Attention ###################
############################################################

# Simplified embedding vector representation
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
print(f'inputs.shape: {inputs.shape}')
# Simplified attention score computation
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
print(attn_scores_2)

for i, x_i in enumerate(inputs):
	attn_scores_2[i] = torch.dot(x_i, query)

print('-----')
print('attn_scores_2:')
print(attn_scores_2)
# tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])


# Reminder: a dot product multiplies two vectors 
# elements-wise and sums the resulting products:

res_0_and_2 = 0. # example for comuting attention score of 0 in regards to the query (input[1])

for idx, element in enumerate(inputs[0]):
    res_0_and_2 += inputs[0][idx] * query[idx]

print('inputs[0]:')
print(inputs[0])
print('query:')
print(query)
print(f'res_0_and_2: {res_0_and_2}')
print(torch.dot(inputs[0], query))



# Simplified attention score normalization
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# Naive softmax version
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Naive Softmax Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# Optimized PyTorch version
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Torch Softmax Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# Compute the final context vector for 2nd input vector
query = inputs[1] # 2nd input token is the query

# We multiply every embedding with its att_weight and compute the sum of all these multiplications
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
    print(f'{i}: {context_vec_2}')
print('-----')
print('context_vec_2:')
print(context_vec_2)

#############################################################################################################
################## Simplified Attention : Computing attention weights for all input tokens ##################
#############################################################################################################

# 1. Compute attention scores (simplified representation = dot product of query with every other x(i))
# 2. Compute attention weights (simplified representation = normalized version of the attention scores (with softmax))
# 3. Compute context vectors (simplified representation = sum of each x(i) * w(i) FOR EACH INPUT)

# Apply previous **step 1** to all pairwise elements to compute the unnormalized attention score matrix:

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

attn_scores = torch.empty(6, 6)


# For each input x_i we create a matrix of 6 attention_scores corresponding to the 6 inputs x_j
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print('-----')
print('attn_score:')
print(attn_scores)

# We can achieve the same as above more efficiently via matrix multiplication:
attn_scores = inputs @ inputs.T
print('-----')
print('attn_score:')
print(attn_scores)

# Similar to step 2 previously, we normaliwe each row so that the values in each row sum to 1:
attn_weights = torch.softmax(attn_scores, dim=1)
print('-----')
print('attn_weights:')
print(attn_weights)

# Verification
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print('-----')
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# Step 3 : compute all context vectors (simplified representation = (sum of each x(i) * w(i)) that we multiply with x_(i) FOR EACH INPUT)
all_context_vecs = attn_weights @ inputs
print('-----')
print('all_context_vecs:')
print(all_context_vecs)

# As a for loop it would be :
all_context_vecs = torch.zeros_like(inputs)  # (6,3)
for i in range(attn_weights.shape[0]):   # pour chaque token i
    context_vec = torch.zeros(inputs.shape[1])  # vecteur (3,)
    for j in range(inputs.shape[0]):     # pour chaque token j
        # print(f'attn_weights[{i}, {j}]: {attn_weights[i, j]}')
        context_vec += attn_weights[i, j] * inputs[j]
        # exit()
    all_context_vecs[i] = context_vec

print("Previous 2nd context vector:", context_vec_2)

##################################################################################################################
################## Real Attention Implementation : Computing the attention weights step by step ##################
##################################################################################################################

# - In GPT models, the input and output dimensions are usually the same, but for illustration purposes, to better follow the computation, we choose different input and output dimensions here:

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# /!\ On a un seul tenseur w_query, W_key, W_value pour touts les inputs
# Dans l'equation de l'attention les valeurs 'Q', 'K' et 'V' représentent en fait respectivement : 
# 'W_q * embedding de l'input', 'W_k * embedding de l'input', 'W_v * embedding de l'input'

print('-----')
print('W_query:')
print(W_query)

# Next we compute the query, key, and value vectors:

query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value

print('-----')
print('x_2:')
print(x_2)

print('-----')
print('query_2:')
print(query_2)

# As we can see below, we successfully projected the 6 input tokens from a 3D onto a 2D embedding space:
# (All this part is in regards to input 2 (?)) -> i don't think so
keys = inputs @ W_key 
values = inputs @ W_value

print('-----')
print('keys:')
print(keys)
print('values:')
print(values)
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
# Les vecteurs keys et values sont de shape (6, 2) 
# >>> une ligne pour (chaque mot de l') input
# >>> une colonne pour chaque dimension de keys

# In the next step, **step 2**, we compute the unnormalized attention scores by computing the dot product between the query and each key vector
# > on a toujours le meme vecteur de query (celui de l'input 2 a laquelle on s'interesse actuellement) que l'on dot product avec chaque key vector correspondant a chaque input

# Chaque mot correspond a un embedding
# La matrice Wq est la meme pour tout le monde 

keys_2 = keys[1]
print('-----')
print('key_2:')
print(key_2)

# Ici on obtient l'attention score de l'input 2 vis a vis de lui meme en faisant le dot product de son vecteur query avec son vecteur key (qui a été obtenu en multipliant son embedding avec le vecteur de weight de keys : W_key) 
attn_score_22 = query_2.dot(keys_2)
print('attn_score_22:')
print(attn_score_22)

# Since we have 6 inputs, we have 6 attention scores for the given query vector:
# Ici on obtient tous les attention score relatifs à l'input 2, en faisant la multiplication matricielle du vecteur de query de l'input 2 AVEC la matrice keys (obtenue en multipliant la matrice inputs avec la matrice W_key)
# On obtient donc ici le resultat de la partie QK (relative a l'input 2) dans l'equation de l'attention.
# Soit tous les attentions scores relatif à l'input 2
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print('attn_score_2:')
print(attn_scores_2)

# On applique maintenant la suite de l'equation : la division par la racine carrée de dk puis le passage du résultat dans softmax
# Next, in **step 3**, we compute the attention weights (normalized attention scores that sum up to 1) using the softmax function we used earlier
# The difference to earlier is that we now scale the attention scores by dividing them by the square root of the embedding dimension, $\sqrt{d_k}$ (i.e., `d_k**0.5`):


print('-----')
d_k = keys.shape[1]
print(f'd_k: {d_k}')
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print('attn_weights_2:')
print(attn_weights_2)
# Prints : tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])
# Ceci est donc la matrice des attention weights de l'input 2 (chaque valeur represente le pourcentage d'attention de l'input vis a vis de l'input 2)

# Il faut maintenant les multiplier avec les values (rappel: values est le tenseur resultant de la multiplication des embeddings par le tenseur W_values) - ET faire l'addition de chaque vecteur de attention_weight * value - pour obtenir le context vector final
# (From Schema) The last step is multiplying each value vector with its respective attention weight and then summing them to obtain the context vector (which will have the shape of the output dimension : [2])
context_vec_2 = attn_weights_2 @ values
print('-----')
print('context_vec_2:')
print(context_vec_2)

print(f'attn_weights_2.shape: {attn_weights_2.shape}')
print(f'values.shape: {values.shape}')
print(f'context_vec_2.shape: {context_vec_2.shape}')

# Prints:
# context_vec_2:
# tensor([0.3061, 0.8210])
# attn_weights_2.shape: torch.Size([6])
# values.shape: torch.Size([6, 2])
# context_vec_2.shape: torch.Size([2])

# De cette maniere les values pour chaque token sont 'ponderees' par l'attention qui doit leur etre portée (vis à vis de l'input 2) puis on additionne tous les resultats pour obtenir un vecteur de taille [2] synthetique, representatif du context pour l'input 2.

# Intuition
# Chaque token est un document résumé en 2 nombres (values en 2D).
# L’attention calcule “combien le token 2 regarde chaque document” (attn_weights_2, 6 poids).
# Puis elle fait une moyenne pondérée de ces résumés → ça donne un seul vecteur de dimension 2 (context_vec_2).

###############################################################################
################# Implementing a compact Self Attention Class #################
###############################################################################

import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print('-----')
print('sa_v1:')
print('inputs:')
print(inputs)
print(sa_v1(inputs))
# -----
# sa_v1:
# tensor([[0.2996, 0.8053],
#         [0.3061, 0.8210],
#         [0.3058, 0.8203],
#         [0.2948, 0.7939],
#         [0.2927, 0.7891],
#         [0.2990, 0.8040]], grad_fn=<MmBackward0>)


# Version using nn.Linear :
# - We can streamline the implementation above using PyTorch's Linear layers, which are equivalent to a matrix multiplication if we disable the bias units
# - Another big advantage of using `nn.Linear` over our manual `nn.Parameter(torch.rand(...)` approach is that `nn.Linear` has a preferred weight initialization scheme, which leads to more stable model training

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

    # def forward(self, x):
    #     keys = self.W_key(x)
    #     queries = self.W_query(x)
    #     values = self.W_value(x)
    #     print('keys:')
    #     print(keys)
        
    #     attn_scores = queries @ keys.T
    #     print('attn_scores:')
    #     print(attn_scores)
    #     attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    #     print('attn_weights:')
    #     print(attn_weights)

    #     context_vec = attn_weights @ values
    #     print('context_vec:')
    #     print(context_vec)
    #     return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print('-----')
print('inputs:')
print(inputs)
context_vec = sa_v2(inputs)
print('sa_v2 context vec:')
print(context_vec)

# -----
# sa_v2:
# tensor([[-0.5337, -0.1051],
#         [-0.5323, -0.1080],
#         [-0.5323, -0.1079],
#         [-0.5297, -0.1076],
#         [-0.5311, -0.1066],
#         [-0.5299, -0.1081]], grad_fn=<MmBackward0>)
# The outputs of v1 and v2 are different because the weights are initiated differently

####################################################################### 
############## Hiding future words with causal attention ##############

############################################################## 
############## Applying a causal attention mask ##############
############################################################## 

# [...] Explaining how we implement and why using -inf mask is the best option

print('-----')
context_length = attn_scores.shape[0]
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print('mask:')
print(mask)
attn_scores_masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print('attn_scores_masked:')
print(attn_scores_masked)
attn_weights_masked = torch.softmax(attn_scores_masked / keys.shape[-1]**0.5, dim=-1)
print('attn_weights_masked:')
print(attn_weights_masked)


########################################################################
############# Implementing a compact causal self-attention #############
########################################################################

# - Now, we are ready to implement a working implementation of self-attention, including the causal and dropout masks
# - One more thing is to implement the code to handle batches consisting of more than one input so that our `CausalAttention` class supports the batch outputs produced by the data loader we implemented in chapter 2
# - For simplicity, to simulate such batch input, we duplicate the input text example

print('-----')
batch = torch.stack((inputs, inputs), dim=0)
print('batch.shape:')
print(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3 
print(batch)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # new
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape # new batch dimension b
        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forward method. 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2) # Changed transpose (because of batch (?))
        # attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        print('*****')
        print('before dropout:')
        print(attn_weights)

        attn_weights = self.dropout(attn_weights) # New

        print('*****')
        print('after dropout:')
        print(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.5)
context_vecs = ca(batch)

print('-----')
print('context_vecs:')
print(context_vecs)
print("context_vecs.shape\n", context_vecs.shape)

# *****
# before dropout:
# tensor([[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.4833, 0.5167, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.3190, 0.3408, 0.3402, 0.0000, 0.0000, 0.0000],
#          [0.2445, 0.2545, 0.2542, 0.2468, 0.0000, 0.0000],
#          [0.1994, 0.2060, 0.2058, 0.1935, 0.1953, 0.0000],
#          [0.1624, 0.1709, 0.1706, 0.1654, 0.1625, 0.1682]],

#         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.4833, 0.5167, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.3190, 0.3408, 0.3402, 0.0000, 0.0000, 0.0000],
#          [0.2445, 0.2545, 0.2542, 0.2468, 0.0000, 0.0000],
#          [0.1994, 0.2060, 0.2058, 0.1935, 0.1953, 0.0000],
#          [0.1624, 0.1709, 0.1706, 0.1654, 0.1625, 0.1682]]],
#        grad_fn=<SoftmaxBackward0>)
# *****
# after dropout:
# tensor([[[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.9665, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.6804, 0.0000, 0.0000, 0.0000],
#          [0.4889, 0.0000, 0.5085, 0.0000, 0.0000, 0.0000],
#          [0.3988, 0.4120, 0.0000, 0.3869, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3363]],

#         [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.9665, 1.0335, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.6380, 0.0000, 0.6804, 0.0000, 0.0000, 0.0000],
#          [0.0000, 0.5090, 0.5085, 0.4936, 0.0000, 0.0000],
#          [0.3988, 0.4120, 0.4116, 0.0000, 0.0000, 0.0000],
#          [0.3249, 0.3418, 0.0000, 0.3308, 0.3249, 0.0000]]],
#        grad_fn=<MulBackward0>)
# -----
# context_vecs:
# tensor([[[-0.9038,  0.4432],
#          [-0.4368,  0.2142],
#          [-0.4849, -0.1341],
#          [-0.5834,  0.0081],
#          [-0.6219, -0.0526],
#          [-0.1417, -0.0505]],

#         [[ 0.0000,  0.0000],
#          [-1.1749,  0.0116],
#          [-0.7733,  0.0073],
#          [-0.9140, -0.2769],
#          [-0.7679, -0.0735],
#          [-0.6749, -0.0984]]], grad_fn=<UnsafeViewBackward0>)
# context_vecs.shape
#  torch.Size([2, 6, 2])

###################################################################################
############# Extending single-head attention to multi-head attention #############
###################################################################################

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


###################################################################################
############# Extending single-head attention to multi-head attention #############
###################################################################################

# - While the above is an intuitive and fully functional implementation of multi-head attention (wrapping the single-head attention `CausalAttention` implementation from earlier), we can write a stand-alone class called `MultiHeadAttention` to achieve the same
# - We don't concatenate single attention heads for this stand-alone `MultiHeadAttention` class
# - Instead, we create single W_query, W_key, and W_value weight matrices and then split those into individual matrices for each attention head:

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, 
        # this will result in errors in the mask creation further below. 
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forward method.

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# - Note that the above is essentially a rewritten version of `MultiHeadAttentionWrapper` that is more efficient
# - The resulting output looks a bit different since the random weight initializations differ, but both are fully functional implementations that can be used in the GPT class we will implement in the upcoming chapters
# - Note that in addition, we added a linear projection layer (`self.out_proj `) to the `MultiHeadAttention` class above. This is simply a linear transformation that doesn't change the dimensions. It's a standard convention to use such a projection layer in LLM implementation, but it's not strictly necessary (recent research has shown that it can be removed without affecting the modeling performance; see the further reading section at the end of this chapter)

# - Note that if you are interested in a compact and efficient implementation of the above, you can also consider the [`torch.nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) class in PyTorch

# - Since the above implementation may look a bit complex at first glance, let's look at what happens when executing `attn_scores = queries @ keys.transpose(2, 3)`: