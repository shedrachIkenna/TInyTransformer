import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from tqdm import tqdm 
from pathlib import Path
import math

# Hyperparams 
device = torch.device("cpu")
block_size = 64
batch_size = 12


# Data loader 
data_path = Path("data/tiny.txt")
assert data_path.exists(), "Create data/text.txt with some text"
text = data_path.read_text(encoding="utf-8")
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}  #This builds a dictionary (mapping) from each character (ch) to a number (i).
itos = {i:ch for ch,i in stoi.items()} # This does the reverse of stoi: creates a dictionary that maps each number back to a character.

def encode(s):
    """
    This function takes in a string as an input and returns its numerical representation based on the stoi mappings 
    """
    return [stoi[c] for c in s]

def decode(l): 
    """
    Takes in a list of numbers and returns a string based in itos mappings 
    """
    return "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = len(data)
train_data = data[: int(0.9 * n)]
val_data = data[int(0.9 * n):]

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Handle for both even and odd d_model
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].size(1)])

        self.register_buffer('pe', pe)

    def forward(self, x):
        t = x.size(1)
        return x + self.pe[:t, :].unsqueeze(0)


