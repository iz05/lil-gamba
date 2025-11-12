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
def load_model(path, device):
    """Load a trained Lil'Gamba model from disk."""
    checkpoint = torch.load(path, map_location=device)
    args = ModelArgs(**checkpoint['model_args'])
    model = Mamba(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def perplexity(model, data_loader, device):
    # TODO: Implement perplexity calculation
    # YOUR CODE HERE
    raise NotImplementedError()

# TODO other metrics we said we would implement