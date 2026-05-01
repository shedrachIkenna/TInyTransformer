import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from tqdm import tqdm 
from pathlib import Path
import math
import matplotlib.pyplot as plt 

# Hyperparams 
device = torch.device("cpu")
block_size = 64
batch_size = 32
dropout = 0.1
n_embd = 128
n_head = 4
n_layer = 4 
learning_rate = 3e-4
max_iters = 2000
eval_interval = 200


# Data loader 
data_path = Path("data/tiny.txt")
assert data_path.exists(), "Create data/text.txt with some text"
text = data_path.read_text(encoding="utf-8")
chars = sorted(list(set(text)))
vocab_size = len(chars)

# convert text to numbers 
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
    # sample random "batch_size" number of indices between 0 and [len(src) - block_size]
    # Here we have 32 random indices which are starting positions to start sampling from 
    ix = torch.randint(len(src) - block_size, (batch_size,)) 

    # select 64 tokens starting from each of those randomly selected positions from above 
    # This becomes the input features 
    x = torch.stack([src[i:i+block_size] for i in ix])

    # Select the same 64 tokens for each starting position but shifted one token left 
    # This becomes the target feature tokens 
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)


class RotaryEmbedding(nn.Module):
    """
    Precomputes the cosine and sine tables used to rotate Q and K vectors.
 
    Core idea:
        Each query/key vector of dimension d_head is treated as d_head/2
        complex numbers. We rotate each pair by an angle that depends on:
            - the token's position  (pos)
            - the frequency of that pair's "slot" (theta_i)
 
        theta_i = 10000^(-2i / d_head)   for i = 0, 1, ..., d_head/2 - 1
 
        The rotation angle for position pos at slot i is:
            pos * theta_i
 
        This is applied as:
            x_rot = x * cos(pos * theta) + rotate_half(x) * sin(pos * theta)
 
        where rotate_half(x) swaps pairs: [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]
 
    Why RoPE beats additive sinusoidal PE:
        - Position info is baked into Q·K dot products, not the token vectors
        - The dot product q_m · k_n only depends on (m - n), i.e. RELATIVE distance
        - No extra parameters; no position vectors added to residual stream
        - Generalises better to longer sequences than seen during training
    """
    def __init__(self, d_head: int, max_len: int = 4096):
        super().__init__()
        # θ_i = 10000^(-2i / d_head) — one frequency per pair of dimensions
        # shape: (d_head // 2,)
        theta = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2).float() / d_head))
 
        # positions 0, 1, 2, ..., max_len-1 — shape: (max_len,)
        pos = torch.arange(max_len).float()
 
        # outer product -> angles[pos, i] = pos * theta_i
        # shape: (max_len, d_head // 2)
        angles = torch.outer(pos, theta)
 
        # Duplicate each angle for the two elements in each pair:
        # [a, b, c] -> [a, a, b, b, c, c]   shape: (max_len, d_head)
        angles = torch.cat([angles, angles], dim=-1)
 
        # Store cos and sin tables as non-trainable buffers
        self.register_buffer('cos', angles.cos())   # (max_len, d_head)
        self.register_buffer('sin', angles.sin())   # (max_len, d_head)

    @staticmethod
    def rotate_half(x):
        """
        Splits x into two halves along the last dimension and produces
        the "perpendicular" vector needed for the rotation:
 
            x  = [x1, x2]          (each is a block of d_head/2 values)
            ->   [-x2, x1]
 
        This implements the 2D rotation matrix applied to every pair:
            [cos  -sin] [x1]   [x1*cos - x2*sin]
            [sin   cos] [x2] = [x1*sin + x2*cos]
 
        We compute the two terms separately and add them in `apply_rope`.
        """
        half = x.shape[-1] // 2
        x1 = x[..., :half]   # first half  of each head vector
        x2 = x[..., half:]   # second half of each head vector
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rope(self, x, T):
        """
        Rotate tensor x using the precomputed cos/sin tables.
 
        Args:
            x : (B, H, T, d_head)  — query or key tensor
            T : sequence length (used to slice the tables)
 
        Returns:
            x_rot : (B, H, T, d_head) — rotated version of x
 
        The formula:
            x_rot = x * cos + rotate_half(x) * sin
 
        cos/sin are sliced to (T, d_head) then broadcast over (B, H).
        """
        cos = self.cos[:T, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head)
        sin = self.sin[:T, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head)
        return x * cos + self.rotate_half(x) * sin
    

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
        """
        LayerNorm Constructor 
        
        super()
            runs the super class nn.Module constructor 
        
        eps:
            tiny number added inside the sqrt for numerical stability
        
        gamma: 
            learnable parameter gamma initialized to ones (so initial scale = 1)
            Making it an nn.Parameter means it will get gradients and be updated by the optimizer
        
        beta: 
            learnable parameter beta initialized to zeros (so initial shift = 0)
            Making it an nn.Parameter means it will get gradients and be updated by the optimizer
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """
        Normalization Calculation Funtion
            Calculate mean
            Calculate variance 
            Apply normalization formula using the mean and variance initially calculated 
            Apply scale and shift using the learnable parameters gamme and beta and the formula 
        x = token embeddings 
        x.shape = (B, T, D)
        """

        mu = x.mean(-1, keepdim=True) # calculate mean 
        var = x.var(-1, unbiased=True, keepdim=True) # calculate variance 
        x_norm = (x - mu) / torch.sqrt(var + self.eps) # normalize x using the normalization formula 
        return self.gamma * x_norm + self.beta # Return x after scale and shift using the learnable parameter gamma and beta   

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rotary: RotaryEmbedding):
        super().__init__() # initiliaze the nn.Module(super class) internals 
        assert d_model % n_heads == 0 # Ensures that the embedding dimensions(features) can be split equally 
        self.d_model = d_model # embedding dimension (D)
        self.n_heads = n_heads # number of attention heads (H)
        self.d_head = d_model // n_heads # dimensions(features) per attention head (d_k = D/H)

        # Combine Wq, Wk, Wv into one big matrix (layer) - This saves computation 
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False) # The nn.Linear class gives us the implemetaion used to create the one big matrix 
        self.out = nn.Linear(d_model, d_model) # W_out weight that mixes attention heads 
        self.dropout = nn.Dropout(dropout) # Dropout is a regularization technique that prevents overfitting 
        
        self.rotary = rotary # get rope instance 
    
    def forward(self, x, mask=None):
        """
        x: tensor of shape (B, T, D). where: 
            D = self.d_model 
        mask: optional attention mask. ignore this for now 
        """
        B, T, D = x.shape


        # Compute Q, K, V together 
        qkv = self.qkv_proj(x) # Compute Q = x.W_q, K = x.W_k, V = x.W_v. All into one matrix qkv 
        # Split qkv into 3 matrices, each having n_heads with n_head in each n_heads 
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_head) 
        # Setting the 3 matrices above to be Q, K, V 
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]


        # Transpose  to (B, H, T, dh) from (B,T,H,dh)
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        # Apply RoPE to q and k 
        q = self.rotary.apply_rope(q, T)
        k = self.rotary.apply_rope(k, T)


        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
 
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
 
        out = torch.matmul(probs, v)                       
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out(out)

class FeedForward(nn.Module):
    """
    The FeedForward layer adds non-linearity to each token's embedding individually. 
    It allows the model to 
        - learn non-linear decision boundaries 
        - build abstract representations 
        - capture higher order interactions (like logic or compositionality)

    Layer Steps:
        - Expand the dimensionality of each token's embeddings 
        - Apply non-linearity using GELU (Gaussian Error Linear Unit)
        - Project the dimension of each token's embeddings back down to its original dimension
        - Apply regularization (dropout) to aviod overfitting 
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        """
        nn.Sequential stacks the multiple layer steps together by connecting the output of one layer to the input of the next just like a pipeline 
        """
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    This is the main Transformer class that combines all the layers above 
    """
    def __init__(self, d_model, n_heads, d_ff, rotary: RotaryEmbedding, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rotary)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        y = self.ln1(x)
        y = self.attn(y, mask=mask)
        x = x + y 
        z = self.ln2(x)
        z = self.ffn(z)
        x = x + z 
        return x 
    

class TinyTransformerLM(nn.Module):
    """
    This is the complete working encoder-decoder (GPT-like) Transformer. Its does the following 
        - Create embeddings 
        - Add positional encoding 
        - Stach transformer blocks (LayerNorm, MultiHeadSelfAttention, FeedForward)
        - End with prediction for next token 
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, block_size, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model) # Create embeddings of length vocab_size and dimension of d_model 
        self.pos_emb = PositionalEncoding(d_model, max_len=block_size+1) # Positional encoding 
        # Create n_layers of transformer blocks. Example if n_layers = 6, it creates 6 transformer block 
        # Each block takes the output of the previous one and refines it further 
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = LayerNorm(d_model) # Final Normalization Layer 

        self.head = nn.Linear(d_model, vocab_size, bias=False) # For every token, it outputs a probability for the next token
        self.block_size = block_size 
        # Tie the weights of inputs embeddings and the output projection. Same vectors used for encode tokens to embeddings are used to decode them 
        self.head.weight = self.token_emb.weight # Reuse embedding weights for decoding. why? final layer weight is just a transpose of the embedding layer weight
        
        # Casual mask 
        # Creates lower triangular matrix which allows each token to attend only to previous tokens, not future ones
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))

        # Initialize weights 
        # Initialize all weights using small random values to help the model start training smoothly 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Sets initial weight values - why?
            - Remember that every neural network learns through the adjustments of parameters (weights and biases). At the start, we must give them initial values before training begins 
        
        What happens if we initialize them poorly? 
            - Gradients vanish or explode 
            - The network fails to learn meaningful patterns 
            - Training becomes slow or unstable 
        """
        if isinstance(module, nn.Linear): # Checks if current layer is a fully connected layer. fully connected layers also need initial bias values 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: # safe checks if bias is defined already
                torch.nn.init.zeros_(module.bias) # Sets initial bias values to zero 
            elif isinstance(module, nn.Embedding): # Check if current layer is embedding layer - Embedding layers only need initial weight values - no bias 
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # Initialize weights using random values with mean around zero and std of 0.02

    def forward(self, idx, targets=None):
        """
        This function does three things 
            - convert input token ids to embeddings 
            - process them through the transformer layers 
            - project final hidden states -> logits -> loss 
        
        Args: 
            idx: batch of token IDs 
            targets: 
        """

        B, T = idx.size()  # gives us the batch and sequence length 
        assert T <= self.block_size # checks if the sequence length is less than or equal to the pre-defined blocksize 
        tok = self.token_emb(idx) # converts IDs to actual embeddings (B, T, D)]
        x = self.pos_emb(tok) # Adds positional encoding 

        # Use cached casual mask 
        mask = self.causal_mask[:, :, :T, :T] 

        # Apply each transformer layer
        for layer in self.layers: 
            x = layer(x, mask=mask)
        
        x = self.ln_f(x) # Apply final normalization layer 

        # projects each token's final hidden vector into a vector of size vocab_size
        logits = self.head(x)  # Each row of logits gives scores for every vocabulary token

        # Compute loss 
        loss = None
        if targets is not None: 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss 
    
    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generates new tokens given a starting context 

        Args:
            idx: starting token IDs, shape [B, T]
            max_new_tokens: number of new tokens to generate 

        Returns: 
            idx: extented token IDs, shape (B, T + max_new_tokens)
        """

        for _ in range(max_new_tokens):
            # crop context to block_size if its grown too long 
            idx_cond = idx[:, -self.block_size:]

            # forward pass 
            logits, _ = self(idx_cond)

            # get the logits of the last step (from the last token, the logits for the next token is computed)
            logits = logits[:, -1, :] # select all batches, select the last step token, select all the vocab scores 

            # convert logits to probabilities 
            probs = F.softmax(logits, dim=-1)

            # Sampling the next token probabilities
            # This way, the model becomes non-deterministic because even though tokens with higher probabilities are likely to be 
            # chosen, lower-probability tokens can also be selected 
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence 
            idx = torch.cat([idx, next_token], dim=1)

# Training Visualization 
def plot_learning_curves(iter_list, train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(iter_list,train_loss, label="Train Loss")
    plt.plot(iter_list, val_loss, label="Val Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

def check_gradient_flow(model):
    """Check the average gradient magnitude per layer"""
    grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and "bais" not in n:
            layers.append(n)
            grads.append(p.grad.abs().mean().item())
    
    plt.figure(figsize=(12, 4))
    plt.bar(layers, grads)
    plt.xticks(rotation=90)
    plt.title("Gradient Magnitude per Layer")
    plt.ylabel("Mean Abs Gradient")
    plt.show()

# Training Execution 
model = TinyTransformerLM(vocab_size, n_embd, n_layer, n_head, n_embd*4, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_history, val_history, iter_history = [], [], [] 

print(f"Training on {device}...")
for i in range(max_iters):
    # Training step 
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Evaluation 
    if i % eval_interval == 0 or i == max_iters - 1: 
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch("val")
            _, v_loss = model(xv, yv)
            print(f"Step {i}: Train Loss {loss.item():.4f}, Val Loss {v_loss.item():.4f}")

            iter_history.append(i)
            train_history.append(loss.item())
            val_history.append(v_loss.item())

        model.train()    

# Save weights after training
torch.save(model.state_dict(), "tiny_transformer.pth")
print("Model weights saved to tiny_transformer.pth")    

# Plot Learning Curves 
plot_learning_curves(iter_history, train_history, val_history)
check_gradient_flow(model)

# Generate something 
print("\nGenerated Sample")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
    
