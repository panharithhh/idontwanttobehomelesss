"""
colab_gpt.py - gpt.py, set up to actually survive Google Colab.

Same architecture and same maths as gpt.py. Four things added, all of which matter
on a free-tier T4 and none of which change the model:

  1. auto-downloads input.txt, so there is nothing to upload
  2. mixed precision - T4 has fp16 tensor cores; this is roughly a 2x speedup.
     fp16 needs a GradScaler, bf16 does not. Detected at runtime, because T4
     (Turing) has NO bf16 while A100/L4 do.
  3. flash attention via F.scaled_dot_product_attention. On CUDA this is a large
     win; on Apple MPS it is worth almost nothing (measured: 358 vs 366 ms/step).
  4. CHECKPOINTING with auto-resume. Free Colab disconnects on idle and caps
     sessions. Re-running this cell picks up where it stopped instead of restarting.

USAGE IN COLAB
  Runtime -> Change runtime type -> T4 GPU     (do this FIRST)

  !wget -q https://raw.githubusercontent.com/panharithhh/idontwanttobehomelesss/main/06-gpt/colab_gpt.py
  !python colab_gpt.py

  ...or just paste this whole file into a cell and run it.

  To keep checkpoints across disconnects, mount Drive first and point CKPT at it:
      from google.colab import drive; drive.mount('/content/drive')
"""

import math, os, time, urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- hyperparameters: Karpathy's full config ----
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

DATA_URL = "https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt"
DATA = "input.txt"
CKPT = "ckpt.pt"          # point at /content/drive/MyDrive/ckpt.pt to survive disconnects
CKPT_EVERY = 500
# -------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(1337)

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"mixed precision: {amp_dtype}")
else:
    print(f"device: {device}  (no CUDA - on Colab do Runtime > Change runtime type > T4 GPU)")
    amp_dtype = None
# fp16 needs loss scaling to stop small gradients flushing to zero; bf16 does not
scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

if not os.path.exists(DATA):
    print(f"downloading {DATA} ...")
    urllib.request.urlretrieve(DATA_URL, DATA)
text = open(DATA, encoding='utf-8').read()
print(f"{DATA}: {len(text):,} characters")

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def ctx():
    """autocast on CUDA, a no-op everywhere else"""
    if amp_dtype is None:
        return torch.autocast(device_type='cpu', enabled=False)
    return torch.autocast(device_type='cuda', dtype=amp_dtype)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx():
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


class MultiheadedAttention(nn.Module):
    """all heads in one batched matmul - same maths as gpt.py's ModuleList of Heads"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        hs = C // n_head
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        q, k, v = [t.view(B, T, n_head, hs).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
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
        x = self.token_embedding_table(idx) + \
            self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B*T, vocab_size), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            with ctx():
                logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :].float(), dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


model = GPTLanguageModel().to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.3f} M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_iter = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optim'])
    start_iter = ck['iter'] + 1
    print(f"resumed from {CKPT} at step {start_iter}")

t0 = time.time()
for it in range(start_iter, max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        el = time.time() - t0
        done = max(it - start_iter, 1)
        eta = el / done * (max_iters - it) / 60
        print(f"step {it:>5}: train {losses['train']:.4f}, val {losses['val']:.4f}"
              f"   [{el/60:.1f} min elapsed, ~{eta:.0f} min left]")

    xb, yb = get_batch('train')
    with ctx():
        _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if it > start_iter and it % CKPT_EVERY == 0:
        torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(),
                    'iter': it}, CKPT)

torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(),
            'iter': max_iters - 1}, CKPT)
print(f"\ndone in {(time.time()-t0)/60:.1f} min, checkpoint saved to {CKPT}\n")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = decode(model.generate(context, 2000)[0].tolist())
print(out)
open('sample.txt', 'w').write(out)
print("\n(2000 more characters written to sample.txt)")
