"""
claude.py - "Attention Is All You Need" (Vaswani et al., 2017), implemented in full.

This is NOT the same architecture as andrej.py. That file is a decoder-only GPT, an
idea from 2018-2020. The 2017 paper is an ENCODER-DECODER built for translation, and
every difference below is a deliberate choice the paper made that modern GPTs reversed:

    paper (this file)                    GPT (andrej.py)
    ------------------------------------ -----------------------------------
    encoder stack + decoder stack        decoder only
    cross-attention decoder->encoder     none
    POST-norm: x = LN(x + Sublayer(x))   PRE-norm: x = x + Sublayer(LN(x))
    sinusoidal position encoding (fixed) learned position embedding
    ReLU in the feed-forward             GELU
    Noam LR schedule with warmup         flat LR
    label smoothing 0.1                  none
    embeddings scaled by sqrt(d_model)   unscaled

Post-norm is the reason the warmup schedule is not optional here. Putting LayerNorm
ON the residual path means early gradients are large and badly conditioned; without
warmup this diverges. That fragility is exactly why the field moved to pre-norm.

THE TASK. The paper translates German to English, so it needs (source, target) pairs.
Shakespeare has no pairs, so we manufacture them: take 2*block_size characters, feed
the first half to the encoder, and have the decoder produce the second half. The model
learns "continue this passage" - which gives cross-attention something real to do
rather than leaving the whole encoder as dead weight.

Paper's base config: d_model=512, N=6, h=8, d_ff=2048, P_drop=0.1, warmup=4000.
Scaled down below to train on a laptop. All structural ratios are preserved.
"""

import math
import time
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------- config ----------------
# paper base: d_model=512, N=6, h=8, d_ff=2048  (~65M params on a 37k vocab)
d_model = 128          # paper's d_model
N = 3                  # paper's N=6 encoder layers AND 6 decoder layers
h = 4                  # paper's h=8 attention heads
d_ff = 512             # paper keeps d_ff = 4 * d_model
dropout_p = 0.1        # paper's P_drop
label_smoothing = 0.1  # paper's eps_ls
warmup = 400           # paper's 4000, scaled to a shorter run

block_size = 64        # source length = target length
batch_size = 32
max_iters = 3000
eval_interval = 300
eval_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# ----------------------------------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
BOS = '\x02'                       # start-of-sequence, fed to the decoder at position 0
itos = dict(enumerate([BOS] + chars))
stoi = {ch: i for i, ch in itos.items()}
vocab_size = len(itos)
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    """src = first block_size chars, tgt = the next block_size chars."""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - 2 * block_size - 1, (batch_size,))
    src = torch.stack([d[i:i+block_size] for i in ix])
    tgt = torch.stack([d[i+block_size:i+2*block_size] for i in ix])
    # decoder input is the target shifted right, with BOS prepended (teacher forcing)
    bos = torch.full((batch_size, 1), stoi[BOS], dtype=torch.long)
    tgt_in = torch.cat([bos, tgt[:, :-1]], dim=1)
    return src.to(device), tgt_in.to(device), tgt.to(device)


def causal_mask(T, dev):
    """(1, 1, T, T) lower-triangular - the paper's 'prevent leftward information flow'."""
    return torch.tril(torch.ones(T, T, device=dev, dtype=torch.bool)).view(1, 1, T, T)


class PositionalEncoding(nn.Module):
    """Section 3.5. Fixed sinusoids, NOT learned.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Chosen so PE(pos+k) is a linear function of PE(pos) - the model can attend by
    relative offset - and so it extrapolates past any length seen in training.
    """

    def __init__(self, d_model, dropout_p, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # not a parameter - never trained

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MultiHeadAttention(nn.Module):
    """Section 3.2.2. One module serving all three uses in the paper:
       encoder self-attention, decoder masked self-attention, and cross-attention.
       The only difference between them is what you pass as q, k, v and mask."""

    def __init__(self, d_model, h, dropout_p):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        # (B, T, D) -> (B, h, T, d_k)
        q = self.w_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)

        # scaled dot-product attention, eq. (1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.w_o(out)


class PositionwiseFeedForward(nn.Module):
    """Section 3.3: FFN(x) = max(0, xW1 + b1)W2 + b2.  ReLU, not GELU."""

    def __init__(self, d_model, d_ff, dropout_p):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """Section 3.1: LayerNorm(x + Dropout(Sublayer(x))).

    POST-norm. The norm is applied AFTER the residual add, so it sits directly on
    the skip path. Modern transformers moved the norm inside the branch instead."""

    def __init__(self, d_model, dropout_p):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """Two sublayers: self-attention, then feed-forward. No mask - the encoder is
       bidirectional, every source position sees every other."""

    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout_p)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
        self.sub = nn.ModuleList([SublayerConnection(d_model, dropout_p) for _ in range(2)])

    def forward(self, x):
        x = self.sub[0](x, lambda y: self.self_attn(y, y, y, None))
        return self.sub[1](x, self.ff)


class DecoderLayer(nn.Module):
    """Three sublayers: masked self-attention, cross-attention over the encoder
       output, then feed-forward. The middle one is what the decoder-only GPT drops."""

    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout_p)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout_p)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
        self.sub = nn.ModuleList([SublayerConnection(d_model, dropout_p) for _ in range(3)])

    def forward(self, x, memory, tgt_mask):
        x = self.sub[0](x, lambda y: self.self_attn(y, y, y, tgt_mask))
        # queries come from the decoder, keys and values from the encoder
        x = self.sub[1](x, lambda y: self.cross_attn(y, memory, memory, None))
        return self.sub[2](x, self.ff)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout_p)
        self.encoder = nn.ModuleList([EncoderLayer() for _ in range(N)])
        self.decoder = nn.ModuleList([DecoderLayer() for _ in range(N)])
        self.generator = nn.Linear(d_model, vocab_size)
        # Section 3.4: share weights between the embedding layers and the pre-softmax
        # linear. Source and target are the same language here, so all three tie.
        self.generator.weight = self.embed.weight
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embed_scaled(self, x):
        # Section 3.4: "we multiply those weights by sqrt(d_model)"
        return self.pos(self.embed(x) * math.sqrt(d_model))

    def encode(self, src):
        x = self.embed_scaled(src)
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, memory, tgt_in):
        x = self.embed_scaled(tgt_in)
        mask = causal_mask(tgt_in.size(1), tgt_in.device)
        for layer in self.decoder:
            x = layer(x, memory, mask)
        return x

    def forward(self, src, tgt_in, tgt=None, smooth=True):
        logits = self.generator(self.decode(self.encode(src), tgt_in))
        if tgt is None:
            return logits, None
        # Section 5.4: label smoothing eps_ls = 0.1. It RAISES the reported loss
        # while improving BLEU, so smoothed loss is not comparable to andrej.py's.
        # Evaluation below reports plain cross-entropy instead.
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1),
                               label_smoothing=label_smoothing if smooth else 0.0)
        return logits, loss

    @torch.no_grad()
    def generate(self, src, max_new_tokens):
        """Autoregressive decoding: encode once, then extend the decoder one step
           at a time. The encoder output is computed a single time and reused."""
        memory = self.encode(src)
        ys = torch.full((src.size(0), 1), stoi[BOS], dtype=torch.long, device=src.device)
        for _ in range(max_new_tokens):
            logits = self.generator(self.decode(memory, ys))
            probs = F.softmax(logits[:, -1, :], dim=-1)
            ys = torch.cat([ys, torch.multinomial(probs, 1)], dim=1)
        return ys[:, 1:]   # drop BOS


def noam_lr(step):
    """Section 5.3:  lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)

    Linear warmup, then inverse-square-root decay. Without the warmup leg, post-norm
    training diverges - this schedule is load-bearing, not a refinement."""
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            src, tgt_in, tgt = get_batch(split)
            _, loss = model(src, tgt_in, tgt, smooth=False)   # unsmoothed = comparable
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


model = Transformer().to(device)
enc_p = sum(p.numel() for p in model.encoder.parameters())
dec_p = sum(p.numel() for p in model.decoder.parameters())
print(f"device: {device}")
print(f"{sum(p.numel() for p in model.parameters())/1e6:.4f} M parameters "
      f"(encoder {enc_p/1e6:.3f}M, decoder {dec_p/1e6:.3f}M)")
print(f"vocab {vocab_size}  |  uniform-guess loss {math.log(vocab_size):.4f}\n")

# Section 5.3: Adam with beta2=0.98 and eps=1e-9, not the usual defaults
optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, noam_lr)

sync()
t0 = time.time()
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter:>5}: train {losses['train']:.4f}, val {losses['val']:.4f}"
              f"   lr {scheduler.get_last_lr()[0]:.2e}")
    src, tgt_in, tgt = get_batch('train')
    _, loss = model(src, tgt_in, tgt)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
sync()
print(f"\ntotal wall time: {time.time()-t0:.1f}s")

# condition on a real passage and let the decoder continue it
src, _, _ = get_batch('val')
src = src[:1]
print("--- SOURCE (fed to the encoder) ---")
print(decode(src[0].tolist()))
print("\n--- DECODER CONTINUATION ---")
print(decode(model.generate(src, 300)[0].tolist()))
