# train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from model import Mamba, ModelArgs
from lilgamba import LilGamba, GambaArgs

# ---------------------------
# 1. Config
# ---------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
SEQ_LEN = 128
EPOCHS = 5
LR = 1e-3
SANITY_CHECK_EVERY = 1  # generate text every n epochs
TEXT_CORPUS = 'miniscule_shakespeare.txt'  # small text dataset

# TODO 1: Get a better text corpus that is still a reasonable size for quick training
# note: we might use tiny_tiny_shakespeare.txt for the draft since this already takes 2 hours to train

# ---------------------------
# 2. Dataset
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokens = tokenizer(text)['input_ids']
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load corpus
with open(TEXT_CORPUS, 'r', encoding='utf-8') as f:
    text_data = f.read()

dataset = TextDataset(text_data, tokenizer, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------
# 3. Model
# ---------------------------
# Use padded vocab size from ModelArgs
# MODEL_NAME = "mamba""
# args = ModelArgs(d_model=64, n_layer=2, vocab_size=len(tokenizer), d_state=16)
# model = Mamba(args).to(DEVICE)
MODEL_NAME = "lilgamba"  # or "Mamba"
args = GambaArgs(d_model=64, n_layer=2, vocab_size=len(tokenizer), d_state=16, num_gamba=4, decay_rate=0.7)
model = LilGamba(args).to(DEVICE)
vocab_size = model.args.vocab_size  # padded vocab size

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# TODO add tracker for losses
# TODO add speed tracker for training

# ---------------------------
# 4. Training loop
# ---------------------------
for epoch in range(1, EPOCHS+1):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
    total_loss = 0.0
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)  # (B, SEQ_LEN, vocab_size)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss/(pbar.n+1))

    # ---------------------------
    # 5. Sanity check text generation
    # ---------------------------
    if epoch % SANITY_CHECK_EVERY == 0:
        model.eval()
        with torch.no_grad():
            seed_text = "ROMEO:"
            input_ids = torch.tensor(tokenizer(seed_text)['input_ids'], dtype=torch.long).unsqueeze(0).to(DEVICE)
            for _ in range(100):
                logits = model(input_ids)
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_id], dim=1)
            generated = tokenizer.decode(input_ids[0].tolist())
            print(f"\n=== Sample Generated Text ===\n{generated}\n============================\n")

# TODO: produce loss plots
# YOUR CODE HERE

# ---------------------------
# 6. Save model
# ---------------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), f"checkpoints/{MODEL_NAME}_tiny.pt")
