"""
tpu.py - gpt.py stripped down and adapted for a Colab TPU (v5e-1).

COLAB SETUP
    Runtime > Change runtime type > v5e-1 TPU > Save
    !pip install -q torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
    !wget -q https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt
    !python tpu.py

WHAT CHANGED FROM gpt.py, AND WHY

  1. device = xm.xla_device(). torch.cuda.is_available() is False on a TPU, so the
     original device line silently falls through to CPU - the exact trap that makes
     people think their TPU is slow.

  2. XLA IS LAZY. Operations queue into a graph and only execute at a sync point.
     xm.optimizer_step(optimizer) is that sync point; without it nothing runs and
     memory grows until the process dies.

  3. FIXED SHAPES OR DEATH. XLA compiles a fresh graph for every distinct tensor
     shape it sees. gpt.py's generate() grows idx by one token per step, so on a TPU
     it would recompile the entire model on EVERY generated token. Here generation
     uses a fixed (1, block_size) rolling window instead - one graph, reused.

  4. No .item() inside the training loop. Every .item() forces a host sync and
     stalls the pipeline. Losses are kept on device and read only at eval time.

  5. bfloat16 autocast. TPUs are built around bf16 natively (unlike a T4, which has
     fp16 tensor cores and no bf16 at all), and it needs no GradScaler.

REMOVED FROM gpt.py: the per-head Head module and the ModuleList in
MultiheadedAttention. Looping over heads in Python emits n_head times as many ops
into the XLA graph for identical maths. One batched projection compiles far better.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch_xla.core.xla_model as xm
    XLA = True
except ImportError:
    XLA = False
    print("torch_xla not found - falling back to cuda/mps/cpu")

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

if XLA:
    device = xm.xla_device()
    autocast_dev, autocast_dtype = 'xla', torch.bfloat16
else:
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    autocast_dev, autocast_dtype = None, None
print(f"device: {device}")

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


def ctx():
    if autocast_dev is None:
        return torch.autocast(device_type='cpu', enabled=False)
    return torch.autocast(device_type=autocast_dev, dtype=autocast_dtype)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        # accumulate on device; one host sync at the end instead of eval_iters of them
        total = torch.zeros((), device=device)
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            with ctx():
                _, loss = model(X, Y)
            total += loss.detach()
        out[split] = (total / eval_iters).item()
    model.train()
    return out


class MultiheadedAttention(nn.Module):
    """all heads in one batched matmul - same maths as gpt.py, far fewer XLA ops"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))
                                          .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        hs = C // n_head
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        q, k, v = [t.view(B, T, n_head, hs).transpose(1, 2) for t in (q, k, v)]
        # explicit attention maths rather than scaled_dot_product_attention: the
        # fused kernel has no portable TPU lowering, and XLA fuses this fine anyway
        wei = (q @ k.transpose(-2, -1)) * hs ** -0.5
        wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        wei = self.dropout(F.softmax(wei, dim=-1))
        out = (wei @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ communication (attention) followed by computation (feedforward) """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadedAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B*T, vocab_size), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, start_id, max_new_tokens):
        """Fixed-shape generation: the window is ALWAYS (1, block_size), so XLA
        compiles one graph and reuses it. Growing the tensor instead would force a
        recompile per token. The window starts zero-padded on the left; that padding
        scrolls out after block_size steps."""
        buf = torch.zeros((1, block_size), dtype=torch.long, device=device)
        buf[0, -1] = start_id
        out = [start_id]
        for _ in range(max_new_tokens):
            with ctx():
                logits, _ = self(buf)
            probs = F.softmax(logits[:, -1, :].float(), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            out.append(nxt.item())
            buf = torch.cat([buf[:, 1:], nxt], dim=1)
        return out


model = GPTLanguageModel().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    with ctx():
        logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if XLA:
        # executes the queued graph AND steps the optimizer. plain optimizer.step()
        # would queue more work without ever running it
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()

print(decode(model.generate(0, 500)))
