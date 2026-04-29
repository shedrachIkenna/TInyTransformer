"""
Sampling Strategy Comparison for GPT-Style Language Model Training

Research Project — MSc Level
Author : Shedrach
Date   : 2025

Research Question
-----------------
Does the method used to sample training batches affect the speed of
convergence and final loss of a character-level transformer language
model trained for a fixed number of iterations?

Four strategies are compared under identical conditions:

    1. Random with replacement   (baseline)
    2. Shuffle without replacement  (epoch-based coverage)
    3. Circular buffer           (wrap-around indexing)
    4. Circular + shuffle        (full coverage + no repetition)

Methodology
-----------
- All four runs start from the SAME initial model weights (saved before
  any training begins and reloaded before each run).
- All four runs use the SAME random seed for reproducibility.
- Training is fixed at max_iters steps for a fair wall-clock comparison.
- Validation loss is averaged over eval_batches batches at each checkpoint
  to reduce noise (single-batch val loss is unreliable for research).
- Results are saved to CSV and plotted at the end.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import csv
import math
import random
from copy import deepcopy
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 0.  REPRODUCIBILITY
# =============================================================================

SEED = 42


def set_seed(seed: int) -> None:
    """Fix all random sources for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =============================================================================
# 1.  HYPERPARAMETERS
# =============================================================================

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 64
batch_size = 32
dropout    = 0.1
n_embd     = 128
n_head     = 4
n_layer    = 4
lr         = 3e-4
max_iters  = 2000

eval_interval = 100  # record loss every N steps
eval_batches  = 20   # average val loss over this many batches (research rigour)

# Loss threshold used to measure convergence speed
CONVERGENCE_THRESHOLD = 1.5

print(f"Device : {device}")
print(f"Seed   : {SEED}")
print(f"Iters  : {max_iters}  |  Eval every {eval_interval} steps")
print(f"Val loss averaged over {eval_batches} batches per checkpoint")
print("=" * 60)


# =============================================================================
# 2.  DATA LOADING
# =============================================================================

data_path = Path("/content/tiny.txt")
assert data_path.exists(), (
    "tiny.txt not found. Upload your text file to /content/tiny.txt"
)

text       = data_path.read_text(encoding="utf-8")
chars      = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(lst: list[int]) -> str:
    return "".join(itos[i] for i in lst)


data       = torch.tensor(encode(text), dtype=torch.long)
n          = len(data)
train_data = data[: int(0.9 * n)]
val_data   = data[int(0.9 * n) :]

print(f"Corpus  : {n:,} tokens")
print(f"Vocab   : {vocab_size} unique characters")
print(f"Train   : {len(train_data):,} tokens")
print(f"Val     : {len(val_data):,} tokens")
print(f"Valid starting positions (train): {len(train_data) - block_size:,}")
print("=" * 60)


# =============================================================================
# 3.  SAMPLING STRATEGIES
# =============================================================================

# ── Strategy 1: Random with replacement (baseline) ───────────────────────────

def get_batch_random(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Original approach. Samples batch_size random starting indices
    uniformly from [0, len(src) - block_size - 1].

    Same index can appear multiple times in the same batch or across
    different steps. No coverage guarantee.
    """
    src = train_data if split == "train" else val_data
    ix  = torch.randint(len(src) - block_size, (batch_size,))
    x   = torch.stack([src[i : i + block_size]         for i in ix])
    y   = torch.stack([src[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ── Strategy 2: Shuffle without replacement ───────────────────────────────────

class ShuffleSampler:
    """
    Epoch-based sampler. Precomputes all valid starting positions,
    shuffles them, then yields batches in order.

    When all positions are exhausted, reshuffles and starts a new epoch.
    Every position is seen exactly once per epoch — no repetition.
    """

    def __init__(self, src: torch.Tensor) -> None:
        self.src     = src
        self.n_valid = len(src) - block_size
        self._reset()

    def _reset(self) -> None:
        perm       = torch.randperm(self.n_valid)
        self.queue = perm.tolist()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.queue) < batch_size:
            self._reset()  # start a new epoch
        ix         = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        ix         = torch.tensor(ix, dtype=torch.long)
        x          = torch.stack([self.src[i : i + block_size]         for i in ix])
        y          = torch.stack([self.src[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


# ── Strategy 3: Circular buffer ───────────────────────────────────────────────

def get_batch_circular(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Treats the data as a ring. Every index in [0, len(src) - 1] is a
    valid starting position. Indices near the end wrap back to the
    beginning using modular arithmetic.

    Trade-off: occasionally produces a context window that stitches the
    end of the corpus to the beginning — a semantically artificial join.
    """
    src = train_data if split == "train" else val_data
    N   = len(src)
    ix  = torch.randint(N, (batch_size,))
    x   = torch.stack([src[torch.arange(i,     i + block_size)     % N] for i in ix])
    y   = torch.stack([src[torch.arange(i + 1, i + block_size + 1) % N] for i in ix])
    return x.to(device), y.to(device)


# ── Strategy 4: Circular + shuffle (hybrid) ───────────────────────────────────

class CircularShuffleSampler:
    """
    Combines the benefits of strategies 2 and 3.

    - Full index range [0, len(src) - 1] via circular wrapping.
    - Shuffle without replacement ensures every position is visited
      exactly once per epoch before any index is revisited.
    """

    def __init__(self, src: torch.Tensor) -> None:
        self.src = src
        self.N   = len(src)
        self._reset()

    def _reset(self) -> None:
        perm       = torch.randperm(self.N)
        self.queue = perm.tolist()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.queue) < batch_size:
            self._reset()
        ix         = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        ix         = torch.tensor(ix, dtype=torch.long)
        N          = self.N
        x          = torch.stack([self.src[torch.arange(i,     i + block_size)     % N] for i in ix])
        y          = torch.stack([self.src[torch.arange(i + 1, i + block_size + 1) % N] for i in ix])
        return x.to(device), y.to(device)


# =============================================================================
# 4.  MODEL COMPONENTS
# =============================================================================

# ── Rotary Positional Embedding (RoPE) ───────────────────────────────────────
#
# RoPE encodes position by *rotating* query and key vectors in 2-D subspaces
# rather than adding a fixed signal to the token embedding.
#
# For a head dimension d_head, pair consecutive dimensions (0,1), (2,3), …
# Each pair i is rotated by angle  θ_i = pos / 10000^(2i / d_head).
# Because rotation is a linear operation, the inner product q·k depends only
# on the *relative* offset (pos_q − pos_k), giving RoPE its relative-position
# generalisation property.
#
# Implementation note
# -------------------
# The cos/sin tables are precomputed once up to max_seq_len and cached as
# non-learnable buffers.  apply_rotary_emb slices the right prefix at
# runtime, so sequences shorter than max_seq_len are handled with no waste.

class RotaryEmbedding(nn.Module):
    """
    Precomputes and caches the cos/sin rotation tables for RoPE.

    Parameters
    ----------
    d_head    : dimension of a single attention head  (must be even)
    max_seq   : maximum sequence length to precompute
    base      : frequency base (default 10 000, as in the original paper)
    """

    def __init__(self, d_head: int, max_seq: int = 2048, base: int = 10_000) -> None:
        super().__init__()
        assert d_head % 2 == 0, "RoPE requires an even head dimension"
        # θ_i = 1 / base^(2i / d_head)  for i = 0, 1, …, d_head/2 − 1
        # Shape: (d_head/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute tables up to max_seq  →  (max_seq, d_head/2)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int) -> None:
        pos   = torch.arange(seq_len, dtype=self.inv_freq.dtype,
                             device=self.inv_freq.device)        # (T,)
        freqs = torch.outer(pos, self.inv_freq)                  # (T, d_head/2)
        # Concatenate so that cos/sin cover all d_head dimensions
        emb   = torch.cat([freqs, freqs], dim=-1)                # (T, d_head)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin tables sliced to the actual sequence length."""
        if seq_len > self.cos_cache.size(0):
            self._build_cache(seq_len)
        # Shapes returned: (1, 1, T, d_head) — ready to broadcast over (B, H, T, d_head)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the second half of the last dimension into the first half with
    a sign flip — the algebraic trick that implements 2-D rotation using
    a single matrix multiply.

    For x = [x1, x2]  (split along last dim):
        _rotate_half(x) = [−x2, x1]

    Combined with cos/sin:
        x * cos + _rotate_half(x) * sin
        = [x1·cos − x2·sin,  x2·cos + x1·sin]
        = R(θ) · [x1, x2]   (standard 2-D rotation matrix)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Parameters
    ----------
    q, k : shape (B, n_heads, T, d_head)
    cos  : shape (1, 1,       T, d_head)  — from RotaryEmbedding.forward
    sin  : shape (1, 1,       T, d_head)

    Returns
    -------
    q_rot, k_rot : same shape as q, k
    """
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu     = x.mean(-1, keepdim=True)
        var    = x.var(-1, unbiased=True, keepdim=True)
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        rotary_emb: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.qkv_proj  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out       = nn.Linear(d_model, d_model)
        self.dropout   = nn.Dropout(dropout)
        self.rotary_emb = rotary_emb   # shared across all layers

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        qkv     = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # (B, T, n_heads, d_head) → (B, n_heads, T, d_head)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # ── Apply RoPE to q and k (v is left unrotated) ──────────────────
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(T)
            q, k     = apply_rotary_emb(q, k, cos, sin)
        # ── Scaled dot-product attention ─────────────────────────────────
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        probs = self.dropout(F.softmax(scores, dim=-1))
        out   = torch.matmul(probs, v)
        out   = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out(out)


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        rotary_emb: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.ln1  = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, rotary_emb)
        self.ln2  = LayerNorm(d_model)
        self.ffn  = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        block_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb  = nn.Embedding(vocab_size, d_model)
        # RoPE: one shared instance whose cos/sin tables are reused by every
        # layer — no extra parameters, and weight-tying across layers is free.
        d_head          = d_model // n_heads
        self.rotary_emb = RotaryEmbedding(d_head, max_seq=block_size)
        self.layers     = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, self.rotary_emb)
            for _ in range(n_layers)
        ])
        self.ln_f       = LayerNorm(d_model)
        self.head       = nn.Linear(d_model, vocab_size, bias=False)
        self.block_size = block_size

        # Weight tying
        self.head.weight = self.token_emb.weight

        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T   = idx.size()
        assert T <= self.block_size
        # Token embeddings only — RoPE is applied inside each attention layer
        x      = self.token_emb(idx)
        mask   = self.causal_mask[:, :, :T, :T]
        for layer in self.layers:
            x = layer(x, mask=mask)
        logits = self.head(self.ln_f(x))
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            probs     = F.softmax(logits[:, -1, :], dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, next_tok], dim=1)
        return idx


# =============================================================================
# 5.  EVALUATION  (averaged over multiple batches)
# =============================================================================

@torch.no_grad()
def evaluate(model: TinyTransformerLM, n_batches: int = eval_batches) -> dict:
    """
    Compute mean train and val loss over n_batches batches.

    Using multiple batches gives a much more stable loss estimate than a
    single batch — essential for research-quality reporting.
    """
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(n_batches):
            xb, yb = get_batch_random(split)   # random is fine for eval
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


# =============================================================================
# 6.  TRAINING LOOP
# =============================================================================

def train_run(technique_name: str, get_train_batch_fn) -> dict:
    """
    Run a full training loop for one sampling technique.

    Parameters
    ----------
    technique_name      : label used in plots and logs
    get_train_batch_fn  : callable() → (x, y) tensors for training

    Returns
    -------
    dict with keys:
        iter_history, train_history, val_history,
        final_train_loss, final_val_loss,
        convergence_step  (step where val loss first ≤ CONVERGENCE_THRESHOLD,
                           or None if threshold was never reached)
    """
    print(f"\n{'═' * 60}")
    print(f"  Run: {technique_name}")
    print(f"{'═' * 60}")

    # Reload identical initial weights for a fair comparison
    set_seed(SEED)
    model = TinyTransformerLM(
        vocab_size, n_embd, n_layer, n_head,
        n_embd * 4, block_size, dropout,
    ).to(device)
    model.load_state_dict(torch.load("init_weights.pth", map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    iter_history     = []
    train_history    = []
    val_history      = []
    convergence_step = None

    for step in range(max_iters):

        # Training step
        xb, yb  = get_train_batch_fn()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Checkpoint
        if step % eval_interval == 0 or step == max_iters - 1:
            metrics = evaluate(model)
            iter_history.append(step)
            train_history.append(metrics["train"])
            val_history.append(metrics["val"])

            if convergence_step is None and metrics["val"] <= CONVERGENCE_THRESHOLD:
                convergence_step = step

            converged_tag = " ← converged" if convergence_step == step else ""
            print(
                f"  Step {step:>4d} | "
                f"Train {metrics['train']:.4f} | "
                f"Val {metrics['val']:.4f}"
                f"{converged_tag}"
            )

    return {
        "iter_history"    : iter_history,
        "train_history"   : train_history,
        "val_history"     : val_history,
        "final_train_loss": train_history[-1],
        "final_val_loss"  : val_history[-1],
        "convergence_step": convergence_step,
    }


# =============================================================================
# 7.  INITIALISE MODEL AND SAVE STARTING WEIGHTS
# =============================================================================

print("\nInitialising model and saving starting weights...")
set_seed(SEED)
_init_model = TinyTransformerLM(
    vocab_size, n_embd, n_layer, n_head,
    n_embd * 4, block_size, dropout,
).to(device)
torch.save(_init_model.state_dict(), "init_weights.pth")
n_params = sum(p.numel() for p in _init_model.parameters() if p.requires_grad)
print(f"Model parameters : {n_params:,}")
print("init_weights.pth saved — all four runs will start from this checkpoint.")
del _init_model


# =============================================================================
# 8.  INSTANTIATE CLASS-BASED SAMPLERS
# =============================================================================

shuffle_sampler          = ShuffleSampler(train_data)
circular_shuffle_sampler = CircularShuffleSampler(train_data)


# =============================================================================
# 9.  RUN ALL FOUR EXPERIMENTS
# =============================================================================

experiments = {
    "Random (baseline)"  : lambda: get_batch_random("train"),
    "Shuffle"            : shuffle_sampler.next_batch,
    "Circular"           : lambda: get_batch_circular("train"),
    "Circular + Shuffle" : circular_shuffle_sampler.next_batch,
}

results = {}
for name, fn in experiments.items():
    results[name] = train_run(name, fn)


# =============================================================================
# 10. SAVE RAW RESULTS TO CSV
# =============================================================================

csv_path = "sampling_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["technique", "step", "train_loss", "val_loss"])
    for name, res in results.items():
        for step, tl, vl in zip(
            res["iter_history"], res["train_history"], res["val_history"]
        ):
            writer.writerow([name, step, f"{tl:.6f}", f"{vl:.6f}"])

print(f"\nRaw results saved to {csv_path}")


# =============================================================================
# 11. SUMMARY TABLE
# =============================================================================

print("\n" + "═" * 65)
print(f"  {'Technique':<25} {'Final Train':>11} {'Final Val':>10} {'Converged @':>12}")
print("─" * 65)
for name, res in results.items():
    conv = (
        f"step {res['convergence_step']}"
        if res["convergence_step"] is not None
        else "not reached"
    )
    print(
        f"  {name:<25} "
        f"{res['final_train_loss']:>11.4f} "
        f"{res['final_val_loss']:>10.4f} "
        f"{conv:>12}"
    )
print("═" * 65)
print(f"  Convergence threshold : val loss ≤ {CONVERGENCE_THRESHOLD}")
print(f"  Val loss averaged over {eval_batches} batches per checkpoint")


# =============================================================================
# 12. PLOTS
# =============================================================================

COLORS = {
    "Random (baseline)"  : "#E24B4A",
    "Shuffle"            : "#1D9E75",
    "Circular"           : "#534AB7",
    "Circular + Shuffle" : "#BA7517",
}

fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "Sampling Strategy Comparison — Character-Level Transformer LM\n"
    f"block_size={block_size}  batch_size={batch_size}  "
    f"n_embd={n_embd}  n_layer={n_layer}  seed={SEED}",
    fontsize=13,
    y=0.98,
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Plot 1: Training loss curves ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for name, res in results.items():
    ax1.plot(
        res["iter_history"], res["train_history"],
        label=name, color=COLORS[name], linewidth=1.8,
    )
ax1.set_title("Training loss", fontsize=11)
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Validation loss curves ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for name, res in results.items():
    ax2.plot(
        res["iter_history"], res["val_history"],
        label=name, color=COLORS[name], linewidth=1.8,
    )
ax2.axhline(
    y=CONVERGENCE_THRESHOLD, color="black",
    linestyle="--", linewidth=1, alpha=0.5,
    label=f"Threshold ({CONVERGENCE_THRESHOLD})",
)
ax2.set_title("Validation loss", fontsize=11)
ax2.set_xlabel("Step")
ax2.set_ylabel("Loss")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── Plot 3: Final train vs val loss (grouped bar chart) ──────────────────────
ax3          = fig.add_subplot(gs[1, 0])
names        = list(results.keys())
short_names  = ["Random", "Shuffle", "Circular", "Circ+Shuf"]
x            = np.arange(len(names))
width        = 0.35
train_vals   = [results[n]["final_train_loss"] for n in names]
val_vals     = [results[n]["final_val_loss"]   for n in names]
bars1 = ax3.bar(
    x - width / 2, train_vals, width, label="Train",
    color=[COLORS[n] for n in names], alpha=0.85,
)
bars2 = ax3.bar(
    x + width / 2, val_vals, width, label="Val",
    color=[COLORS[n] for n in names], alpha=0.45,
    edgecolor=[COLORS[n] for n in names], linewidth=1.5,
)
ax3.set_title("Final loss comparison", fontsize=11)
ax3.set_ylabel("Loss")
ax3.set_xticks(x)
ax3.set_xticklabels(short_names, fontsize=9)
ax3.legend(fontsize=9)
ax3.grid(True, axis="y", alpha=0.3)
for bar in bars1:
    ax3.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7,
    )
for bar in bars2:
    ax3.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7,
    )

# ── Plot 4: Convergence speed ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
conv_steps = [
    results[n]["convergence_step"] if results[n]["convergence_step"] is not None
    else max_iters
    for n in names
]
bars = ax4.bar(short_names, conv_steps, color=[COLORS[n] for n in names], alpha=0.85)
ax4.axhline(
    y=max_iters, color="black", linestyle="--",
    linewidth=1, alpha=0.4, label="Max iterations",
)
ax4.set_title(
    f"Convergence speed\n(step val loss ≤ {CONVERGENCE_THRESHOLD})",
    fontsize=11,
)
ax4.set_ylabel("Step")
ax4.legend(fontsize=9)
ax4.grid(True, axis="y", alpha=0.3)
for bar, step, name in zip(bars, conv_steps, names):
    label = (
        str(step)
        if results[name]["convergence_step"] is not None
        else "N/A"
    )
    ax4.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
        label, ha="center", va="bottom", fontsize=9, fontweight="bold",
    )

plt.savefig("sampling_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to sampling_comparison.png")


# =============================================================================
# 13. TEXT GENERATION SAMPLES  (qualitative check)
# =============================================================================

# NOTE: To properly compare generated text across all four models, save each
# model's weights after training:
#
#     torch.save(model.state_dict(), f"model_{name.replace(' ', '_')}.pth")
#
# Then reload and generate from each. This script keeps peak memory low by
# not holding all four models in RAM simultaneously.

print("\n" + "═" * 60)
print("  Generated text samples (200 tokens each)")
print("  All four models trained from the same initial weights")
print("═" * 60)
print("\nTo compare generated text, re-run each technique individually")
print("and save the model weights with torch.save() after training.")
print("\nResearch run complete.")