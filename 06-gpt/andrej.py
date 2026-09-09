"""
andrej.py - reference implementation, nanoGPT style.

This is the batched multi-head attention Karpathy walks through at ~1:47:00 of
"Let's build GPT". It is the ANSWER KEY. gpt.py is yours. Keep them separate.

The one real difference from gpt.py's MultiheadedAttention:

  gpt.py    nn.ModuleList of n_head separate Head modules. Each head runs its own
            three small matmuls, then the results are torch.cat'd together. That is
            n_head * 3 small matmuls driven by a Python for-loop.

  andrej.py ONE nn.Linear(n_embd, 3*n_embd) produces q, k and v for every head in a
            single matmul. A .view() then splits the channel dimension into heads and
            a .transpose() moves the head dim next to the batch dim, so the attention
            maths runs batched over (B, n_head) at once. Identical maths, far fewer
            kernel launches, no Python loop.

Defaults below match gpt.py exactly so the comparison is apples to apples.
Karpathy's full-scale config is in the comment - expect it to be slow on MPS.
"""

import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters - identical to gpt.py so the A/B is fair
# Karpathy's full config: batch_size=64, block_size=256, n_embd=384, n_head=6,
# n_layer=6, dropout=0.2, learning_rate=3e-4, max_iters=5000
batch_size = 32
block_size = 32
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
use_flash = True   # flip to False to time the hand-written attention path instead
# ------------

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
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


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


def sync():
    """Wait for the GPU so timings are real and not just queue-submission time."""
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


class CausalSelfAttention(nn.Module):
    """ all n_head heads at once, in a single batched matmul """

    def __init__(self):
        super().__init__()
        assert n_embd % n_head == 0
        # one projection producing key, query and value for ALL heads
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection - mixes the heads' results back together before the residual
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size))
                                          .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        hs = C // n_head

        # (B,T,C) -> (B,T,3C) -> three tensors of (B,T,C)
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        # split channels into heads and move the head dim beside batch: (B, nh, T, hs)
        # every op after this is batched over B*nh independent attention problems
        k = k.view(B, T, n_head, hs).transpose(1, 2)
        q = q.view(B, T, n_head, hs).transpose(1, 2)
        v = v.view(B, T, n_head, hs).transpose(1, 2)

        if self.flash:
            # fused kernel - never materialises the (B,nh,T,T) matrix in memory
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=dropout if self.training else 0.0,
                is_causal=True)
        else:
            # the version you wrote by hand, just with a leading head dimension
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C): the concat, done as a reshape
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """ per-token computation. 4x expansion is the GPT-2 convention """

    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """ communication then computation, both wrapped in residuals """

    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        # pre-norm: LayerNorm sits INSIDE the residual branch, so the skip path
        # from input to output is a clean unmodified sum
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)   # final norm before the output head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying: the embedding matrix and the output matrix are the same
        # tensor. Saves vocab_size*n_embd parameters and usually helps.
        self.token_embedding_table.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scale down the residual projections so the residual stream does not grow
        # in variance as depth increases (GPT-2 trick)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GPTLanguageModel()
m = model.to(device)
print(f"device: {device}  flash: {model.blocks[0].attn.flash}")
print(f"{sum(p.numel() for p in m.parameters())/1e6:.4f} M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

sync()
t_start = time.time()
step_times = []

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    t0 = time.time()
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    sync()
    step_times.append(time.time() - t0)

total = time.time() - t_start
# ignore the first 10 steps - they include one-off compilation and allocation costs
steady = step_times[10:] or step_times
ms = sum(steady) / len(steady) * 1000
print(f"\ntotal wall time: {total:.1f}s")
print(f"mean step: {ms:.2f} ms   throughput: {batch_size*block_size/(ms/1000):,.0f} tokens/s")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
