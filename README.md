# MyGPT

## Overview

A from-scratch implementation of an LLM model using the Transformer architecture (multi-head attention), based on the papers "Attention is all you need" (Vaswani et al., 2017) and "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) for GPT-2.

This project explores the GPT-2-style language model architecture by implementing all essential components from scratch: positional embeddings, multi-head attention, Transformer blocks, and text generation.

-----
## Usage / How to run

This project uses `uv` as a Python package manager. To install dependencies:

```bash
uv sync
```

To run the scripts:

```bash
uv run python MyGPT_v1_local-train.py
# or
uv run python MyGPT_v2_pretrained.py
```

-----
## MyGPT_v1_local-train.py

This first version offers a complete local training of the GPT model from scratch. It demonstrates the full training pipeline:

- **Model Initialization**: Creates an untrained GPT model with a 124M parameter architecture
- **Data Preparation**: Loads and prepares training text from "the-verdict.txt"
- **Training**: 10 epochs with train/validation split (90/10)
- **Generation**: Demonstrates text generation before and after training

⚠️ **Important note**: This version is designed for conceptual purposes to demonstrate the training process. It quickly shows overfitting (the model memorizes and repeats excerpts from training text) because:
- It's only trained for 10 epochs
- Training data is limited to a single small text file
- The model has significant capacity but insufficient data to train properly

```
===========================================
=========== UNTRAINED GPT MODEL ===========
===========================================

Untrained model created with the following configuration:
- vocab_size: 50257
- context_length: 256
- emb_dim: 768
- n_heads: 12
- n_layers: 12
- drop_rate: 0.1
- qkv_bias: False
-----
Let's run an inference with our raw untrained model:
Input text:
>>> Every effort moves you
Output text:
>>>  Every effort moves you rentingetic wasnم refres RexMeCHicular stren
-----
As we can see, our untrained model does not produce a good text because... it has not been trained yet!

==============================================
=========== TRAINING OUR GPT MODEL ===========
==============================================

Let's train our model and try again...
Loaded trainging data from: the-verdict.txt
Created training and validation data loaders
Using mps device.
==================================================
Starting Training...
Ep 1 (Step 000000): Train loss 9.817, Val loss 9.924
Ep 1 (Step 000005): Train loss 8.066, Val loss 8.332
>>> Every effort moves you,,,,,,,,,,,,.                                     
Ep 2 (Step 000010): Train loss 6.619, Val loss 7.042
Ep 2 (Step 000015): Train loss 6.046, Val loss 6.596
>>> Every effort moves you, and,, and, and,,,,, and, and,,,,,,,,,,, and,, the,, the, and,, and,,, the, and,,,,,,
[...]
Ep 9 (Step 000075): Train loss 1.059, Val loss 6.251
Ep 9 (Step 000080): Train loss 0.800, Val loss 6.278
>>> Every effort moves you?"  "Yes--quite insensible to the fact with a laugh: "Yes--and by me!"  He laughed again, and threw back the window-curtains, I saw that, and down the room, and now
Ep 10 (Step 000085): Train loss 0.569, Val loss 6.373
============================================
=========== POST TRAINING OUTPUT ===========
============================================
>>> Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I
Training completed in 1.09 minutes.
```

## MyGPT_v2_pretrained.py

This second version uses pre-trained weights downloaded from OpenAI's open-source models (GPT-2).

**GPT-2 Principle**: GPT-2 is a language model trained on a very large corpus of internet text. It's specifically designed for **text completion** (intelligent autocompletion) rather than following instructions like ChatGPT.

**How it works**:
1. Automatically downloads GPT-2 model weights (124M parameters, ≈509 MB)
2. Loads these weights into our from-scratch GPT architecture
3. Enables interactive chat where users can complete sentences

### Usage Example

```bash
$> python ./MyGPT_v2_pretrained.py
MyGPT: Please enter a sentence to complete (enter 'exit' to end chat)
> Every effort moves you
MyGPT: Every effort moves you toward a goal. That's the key to success. That's the reason we're always trying to make sure our teams are the best team on the field," he said.
```

This version is ready to use for generating coherent text and creating text completion applications.