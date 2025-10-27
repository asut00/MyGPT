# MyGPT

A from-scratch implementation of an LLM model using the Transformer architecture (multi-head attention), based on the papers "Attention is all you need" (Vaswani et al., 2017) and "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) for GPT-2.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
  - [MyGPT_v1_local-train.py](#mygpt_v1_local-trainpy)
  - [MyGPT_v2_pretrained.py](#mygpt_v2_pretrainedpy)

## Overview

This project explores the GPT-2-style language model architecture by implementing all essential components from scratch: positional embeddings, multi-head attention, Transformer blocks, and text generation.

The implementation includes:
- Multi-head self-attention mechanism
- Positional encodings
- Layer normalization and residual connections
- Transformer decoder blocks
- GPT-2 architecture (12 layers, 12 heads, 768 embedding dimensions)
- Training pipeline with validation
- Text generation with various sampling strategies

## Features

- **From-scratch implementation**: All components built without relying on high-level frameworks
- **Two versions**: Local training from scratch vs. using pre-trained GPT-2 weights
- **Apple Silicon support**: Optimized for MPS (Metal Performance Shaders) on Apple devices
- **Interactive generation**: Chat-like interface for text completion

## Installation

### Prerequisites

- Python 3.8 or higher
- `uv` package manager (recommended) or `pip`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/asut00/MyGPT.git
cd MyGPT
```

2. Install dependencies using `uv`:
```bash
uv sync
```

Or using traditional pip:
```bash
pip install -r requirements.txt
```

The project will automatically download the GPT-2 model weights (≈509 MB) when running `MyGPT_v2_pretrained.py`.

## Usage

This project uses `uv` as a Python package manager. To install dependencies:

```bash
uv sync
```

To run the script with uv:

```bash
uv run python MyGPT_v1_local-train.py
# or
uv run python MyGPT_v2_pretrained.py
```

Or to activate the environment and run the scripts with python:

```bash
source .venv/bin/activate

# then

python MyGPT_v1_local-train.py
# or
python MyGPT_v2_pretrained.py
```

## Project Structure

```
mini-LLM_03/
├── src/                      # Core implementation
│   ├── GPTModel.py          # Main GPT model architecture
│   ├── MultiHeadAttention.py
│   ├── TransformerBlock.py
│   ├── Dataloader.py        # Data loading utilities
│   ├── utils.py             # Text processing utilities
│   ├── utils_train.py       # Training utilities
│   ├── gpt_download.py      # GPT-2 weights downloader
│   └── gpt_loadweights.py   # Weight loading logic
├── 00_walkthrough/          # Learning materials
├── gpt2/124M/               # Pre-trained weights (downloaded automatically)
├── MyGPT_v1_local-train.py  # Local training script
├── MyGPT_v2_pretrained.py   # Pre-trained model script
└── the-verdict.txt          # Training data
```

## Scripts

### MyGPT_v1_local-train.py

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
$> python ./MyGPT_v1_local-train.py
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

==============================================
=========== TRAINING OUR GPT MODEL ===========
==============================================

Loaded training data from: the-verdict.txt
Starting Training...
Ep 1 (Step 000000): Train loss 9.817, Val loss 9.924
Ep 1 (Step 000005): Train loss 8.066, Val loss 8.332
>>> Every effort moves you,,,,,,,,,,,,.                                     
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

### MyGPT_v2_pretrained.py

This second version uses pre-trained weights downloaded from OpenAI's open-source models (GPT-2).

**GPT-2 Principle**: GPT-2 is a language model trained on a very large corpus of internet text. It's specifically designed for **text completion** (intelligent autocompletion) rather than following instructions like ChatGPT.

**How it works**:
1. Automatically downloads GPT-2 model weights (124M parameters, ≈509 MB)
2. Loads these weights into our from-scratch GPT architecture
3. Enables interactive chat where users can complete sentences

### Usage Example

```
$> python ./MyGPT_v2_pretrained.py

============================================
=========== PRETRAINED GPT MODEL ===========
============================================

Downloading model weights for 124 million parameter model (≈509 Mo)...

MyGPT: Please enter a sentence to complete (enter 'exit' to end chat)
> Every effort moves you
MyGPT: Every effort moves you toward a goal. That's the key to success. That's the reason we're always trying to make sure our teams are the best team on the field," he said.
```

This version is ready to use for generating coherent text and creating text completion applications.

## Technical Details

### Model Architecture

The GPT model is based on the Transformer decoder architecture:

- **Vocabulary size**: 50,257 (GPT-2's BPE tokenizer)
- **Embedding dimension**: 768
- **Number of layers**: 12
- **Number of attention heads**: 12
- **Context length**: 256 (training) / 1024 (pretrained)
- **Positional encoding**: Learned embeddings
- **Activation**: GELU

### Key Components

1. **MultiHeadAttention**: Implements scaled dot-product attention with multiple heads
2. **TransformerBlock**: Combines attention with feed-forward layers and layer normalization
3. **GPTModel**: Full model with embedding layers, transformer blocks, and output projection
4. **Text Generation**: Sampling with temperature and top-k filtering for diverse outputs

## Resources

- Based on tutorials by Sebastian Raschka and Andrej Karpathy's nanoGPT
- My raw walkthrough materials are available in `_walkthrough/` directory

## Notes

- The training script (`MyGPT_v1_local-train.py`) uses a shortened context length (256) for faster training on limited hardware
- The pretrained script (`MyGPT_v2_pretrained.py`) uses the full context length (1024) to match GPT-2
- Pre-trained weights are cached in the `gpt2/` directory after first download