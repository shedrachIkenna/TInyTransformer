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

        """
        Calculations for Positional Encoding 
        """
        pe = torch.zeros(max_len, d_model) # Create a zeros matrix of size (max_len X d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Creates a matrix of size (max_len X 1) ex: [0,1,2,3,4...]T 

        # Calculation of the frequency term 
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # Rewrite the formula using exponential and lin identies 

        
        pe[:, 0::2] = torch.sin(pos * div) 
        # Prevents out of bounds error/shape mismatch when embedding dimension has odd number of features(columns)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].size(1)])

        self.register_buffer('pe', pe) # store tensor pe but not as a trainable parameter 

    def forward(self, x): 
        """
            x is a group of sentence embeddings in a 3d tensor 
            x has a size (B=batch(number of sentences), T=sqeuence length(number of tokens per sentence), d=dimension of embeddings)
            x.size(0) = B
            x.size(1) = T
            x.size(2) = d 
        """
        t = x.size(1) 
        """
        
            self.pe[:t, :] - take the first 3 rows of all columns in pe. 
            
        .unsqueeze(0): changes the shape of self.pe[:t, :] 
                        from 
                            (max_len X d_model) 
                        to 
                            (1 X max_len X d_model) 
                    This is done so that elementwise addition can be performed
                    Example: 
                    x                : (B, T, d) = (2, 3, 4)
                    pe[:t, :].unsqueeze(0): (1, T, d) = (1, 3, 4)
        """
        return x + self.pe[:t, :].unsqueeze(0)
    

class LayerNorm(nn.Module):
    """
    Normalization Layer:
        
    Input: 
        Token embeddings x with shape (B,T,D)
    
    Action: 
        Normalizes the dimension D of each token 
        Scales and shift each token's normalized values using learned parameter Gamma and Beta 
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))


