"""
Evaluation utilities for Lil'Gamba.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from model import Mamba, ModelArgs

# TODO: check this implementation

"""
def load_model(path, device):
    ### Load a trained Lil'Gamba model from disk. ###
    checkpoint = torch.load(path, map_location=device)
    args = ModelArgs(**checkpoint['model_args'])
    model = Mamba(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
"""

def load_model(path, device, tokenizer):
    """Load a trained Lil'Gamba model from disk."""
    checkpoint = torch.load(path, map_location=device)

    # args = ModelArgs(**checkpoint['model_args'])

    args = ModelArgs(d_model=64, n_layer=2, vocab_size=len(tokenizer), d_state=16)
    model = Mamba(args).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def perplexity(model, data_loader, device):
    # TODO: Implement perplexity calculation
    # YOUR CODE HERE
    model.eval()

    print("Calculating perplexity of model with test prompts (batched):")
    print("=" * 60)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Calculating Perplexity"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)
            B, T, V = logits.shape

            # Compute cross-entropy loss
            loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T), reduction='none')

            # Update total_loss & total_tokens
            total_loss += loss.sum().item()
            total_tokens += y.numel()   # number of elements

        avg_loss = total_loss / max(1, total_tokens)
        avg_perplexity = float(torch.exp(torch.tensor(avg_loss)))

        print(f"Average Perplexity: {avg_perplexity:.4f}")
        return avg_perplexity

# TODO other metrics we said we would implement
def selective_copy_task(model, device):
  raise NotImplementedError()

def induction_task(model, device):
  raise NotImplementedError()