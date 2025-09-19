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
for i, x_i in enumerate(inputs):
	attn_scores_2[i] = torch.dot(x_i, query)

print('-----')
print('attn_scores_2:')
print(attn_scores_2)
# tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])


# Simplified attention score normalization
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# Naive softmax version
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# Optimized PyTorch version
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# Compute the final context vector for 2nd input vector
query = inputs[1] # 2nd input token is the query

# We multiply every embedding with its att_weight and compute the sum of all these multiplications
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print('-----')
print('context_vec_2:')
print(context_vec_2)

######################################################################################
################## Computing attention weights for all input tokens ##################
######################################################################################

# 1. Compute attention scores (simplified representation = dot product of query with every other x(i))
# 2. Compute attention weights (simplified representation = normalized version of the attention scores (with softmax))
# 3. Compute context vectors (simplified representation = sum of each x(i) * w(i) FOR EACH INPUT)
# ---> then you multiply the input with its context vector ?

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

