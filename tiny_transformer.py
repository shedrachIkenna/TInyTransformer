import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from tqdm import tqdm 
from pathlib import Path

# Hyperparams 
device = torch.device("cpu")


# Data loader 
data_path = Path("data/tiny.txt")
assert data_path.exists(), "Create data/text.txt with some text"
text = data_path.read_text(encoding="utf-8")
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}  #This builds a dictionary (mapping) from each character (ch) to a number (i).
itos = {i:ch for ch,i in stoi.items()} # This does the reverse of stoi: creates a dictionary that maps each number back to a character.


