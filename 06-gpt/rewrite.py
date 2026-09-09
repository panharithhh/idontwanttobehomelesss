import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================================
# HYPERPARAMETERS
# ==========================================
batch_size = 32     # B: how many independent sequences will we process in parallel?
block_size = 32     # T: what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 64         # C: embedding dimension
n_head = 4          # number of attention heads
n_layer = 4         # number of transformer blocks
dropout = 0.0
# ==========================================

torch.manual_seed(1337)

# Load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ==============================================================================
# 1. ONE HEAD OF SELF-ATTENTION
# ==============================================================================
class Head(nn.Module):
    """
    One single head of self-attention.
    - Computes Key, Query, Value projections.
    - Calculates attention scores (wei = Q @ K.T / sqrt(d_k)).
    - Masks future tokens with lower-triangular matrix (tril).
    - Applies softmax + dropout, then multiplies by Value (wei @ V).
    """
    def __init__(self, head_size):
        super().__init__()
        # TODO: Define Linear layers for key, query, value (bias=False)
        # TODO: Register 'tril' buffer for causal masking
        # TODO: Define dropout layer
        raise NotImplementedError

    def forward(self, x):
        # Input x shape: (B, T, C)
        # TODO: Compute k, q, v from x
        # TODO: Compute attention affinities (wei = q @ k.T * head_size**-0.5)
        # TODO: Mask upper triangle with -inf using self.tril
        # TODO: Apply softmax over the last dimension (-1)
        # TODO: Apply dropout to attention weights
        # TODO: Aggregate values (out = wei @ v)
        # Output shape should be (B, T, head_size)
        raise NotImplementedError


# ==============================================================================
# 2. MULTI-HEAD ATTENTION
# ==============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    - Runs `num_heads` in parallel (each with `head_size`).
    - Concatenates all head outputs along the channel dimension.
    - Passes concatenated outputs through a final Linear projection layer.
    - Applies dropout.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        # TODO: Create a ModuleList of `num_heads` Head instances
        # TODO: Define projection Linear layer (n_embd -> n_embd)
        # TODO: Define dropout layer
        raise NotImplementedError

    def forward(self, x):
        # TODO: Run all heads in parallel on x and concatenate on dim=-1
        # TODO: Apply projection layer
        # TODO: Apply dropout
        # Output shape should be (B, T, n_embd)
        raise NotImplementedError


# ==============================================================================
# 3. FEED-FORWARD (COMPUTATION / THINKING)
# ==============================================================================
class FeedForward(nn.Module):
    """
    A simple MLP for per-token computation:
    - Linear projection expanding dimension by 4x: (n_embd -> 4*n_embd)
    - Non-linearity: ReLU
    - Linear projection compressing back: (4*n_embd -> n_embd)
    - Dropout
    """
    def __init__(self, n_embd):
        super().__init__()
        # TODO: Create Sequential: Linear(n_embd, 4*n_embd) -> ReLU -> Linear(4*n_embd, n_embd) -> Dropout
        raise NotImplementedError

    def forward(self, x):
        # TODO: Pass x through the sequential network
        raise NotImplementedError


# ==============================================================================
# 4. TRANSFORMER BLOCK
# ==============================================================================
class Block(nn.Module):
    """
    One Transformer Block:
    - Communication (MultiHeadAttention) followed by Computation (FeedForward).
    - Pre-LayerNorm applied before both sub-layers.
    - Residual connections (x = x + ...) around both sub-layers.
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        # TODO: Calculate head_size = n_embd // n_head
        # TODO: Instantiate MultiHeadAttention
        # TODO: Instantiate FeedForward
        # TODO: Instantiate LayerNorm 1 (for attention)
        # TODO: Instantiate LayerNorm 2 (for feedforward)
        raise NotImplementedError

    def forward(self, x):
        # TODO: 1. Attention step with Pre-LayerNorm and residual connection:
        #          x = x + self.sa(self.ln1(x))
        # TODO: 2. FeedForward step with Pre-LayerNorm and residual connection:
        #          x = x + self.ffwd(self.ln2(x))
        # TODO: Return x
        raise NotImplementedError


# ==============================================================================
# 5. FULL GPT LANGUAGE MODEL
# ==============================================================================
class GPTLanguageModel(nn.Module):
    """
    The Full GPT Model:
    - Token embedding table + Position embedding table
    - Stack of N Transformer Blocks
    - Final LayerNorm
    - Language modeling head (Linear projection to vocab_size)
    """
    def __init__(self):
        super().__init__()
        # TODO: Embedding table for tokens: vocab_size -> n_embd
        # TODO: Embedding table for positions: block_size -> n_embd
        # TODO: Sequential list of `n_layer` Block instances
        # TODO: Final LayerNorm (n_embd)
        # TODO: Final Linear lm_head (n_embd -> vocab_size)
        raise NotImplementedError

    def forward(self, idx, targets=None):
        # Input idx shape: (B, T)
        # TODO: Get token embeddings from token_embedding_table
        # TODO: Get position embeddings from position_embedding_table for torch.arange(T)
        # TODO: Combine x = tok_emb + pos_emb
        # TODO: Pass x through the stack of blocks
        # TODO: Pass x through the final LayerNorm
        # TODO: Pass x through lm_head to get logits: shape (B, T, vocab_size)

        # Compute cross-entropy loss if targets are provided:
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

    def generate(self, idx, max_new_tokens):
        # Autoregressive text generation
        for _ in range(max_new_tokens):
            # Crop current context to the last `block_size` tokens
            idx_cond = idx[:, -block_size:]
            # Get predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample next token from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled token to running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==============================================================================
# TRAINING & GENERATION LOOP
# ==============================================================================
if __name__ == '__main__':
    model = GPTLanguageModel()
    m = model.to(device)
    print(f"{sum(p.numel() for p in m.parameters())/1e6:.4f} M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate sample text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- GENERATED TEXT ---")
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
