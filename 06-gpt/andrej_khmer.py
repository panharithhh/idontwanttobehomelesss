"""
andrej_khmer.py - andrej.py, pointed at Khmer instead of Shakespeare.

The MODEL is byte-identical to andrej.py. Only the data pipeline changed, because
that is where every Khmer-specific problem lives. Three of them:

  1. ZWSP (U+200B). Khmer does not space between words; the zero-width space is
     used to mark word boundaries where they need marking. There are ~32.6k of them
     in this corpus and they are INVISIBLE in your editor and in generated output.
     zwsp_mode below lets you train with and without them - that A/B is the first
     real experiment for the tokenizer project.

  2. A junk vocabulary. khmer_input.txt is already distilled to Khmer script by
     ../07-tokenizer/make_khmer_only.py (884 codepoints -> 95). min_char_freq is
     kept as a second line of defence: it folds any remaining rare codepoint into a
     single UNK rather than giving it its own embedding row and softmax logit.

  3. Loss is NOT comparable to Shakespeare's. Cross-entropy over ~300 classes and
     cross-entropy over 65 classes are different scales. Compare against the
     uniform-baseline line this script prints, never against andrej.py's number.
"""

import math
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------- data config ----------------
data_path = Path(__file__).parent / 'khmer_input.txt'
min_char_freq = 10        # codepoints rarer than this are folded into UNK
zwsp_mode = 'keep'        # 'keep' = train on the word-boundary marks, 'strip' = delete them

# ---------------- model config ----------------
# identical to andrej.py so the only variable is the language
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
use_flash = True
# ---------------------------------------------

ZWSP = '​'
UNK = '�'

torch.manual_seed(1337)

# ---------------- load and clean ----------------
raw = data_path.read_text(encoding='utf-8')
print(f"loaded {data_path.name}: {len(raw):,} chars, {len(set(raw))} unique codepoints")

n_zwsp = raw.count(ZWSP)
if zwsp_mode == 'strip':
    raw = raw.replace(ZWSP, '')
    print(f"zwsp_mode=strip: removed {n_zwsp:,} zero-width spaces")
else:
    print(f"zwsp_mode=keep: {n_zwsp:,} zero-width spaces kept as a real token")

counts = Counter(raw)
keep = {ch for ch, n in counts.items() if n >= min_char_freq}
dropped = len(counts) - len(keep)
n_replaced = sum(n for ch, n in counts.items() if ch not in keep)
text = ''.join(ch if ch in keep else UNK for ch in raw)
print(f"folded {dropped} rare codepoints ({n_replaced:,} chars, "
      f"{n_replaced/len(raw)*100:.3f}% of text) into UNK")

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

khmer = sum(n for ch, n in Counter(text).items() if 'ក' <= ch <= '៿')
print(f"vocab_size {vocab_size}  |  {khmer/len(text)*100:.1f}% Khmer script")
print(f"uniform-guess loss for this vocab: {math.log(vocab_size):.4f} "
      f"(Shakespeare's was {math.log(65):.4f} - do not compare across these)\n")


def show(s):
    """Render invisible characters so you can actually read the samples."""
    return s.replace(ZWSP, '·').replace(UNK, '?')


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


# ---------------- model: unchanged from andrej.py ----------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
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
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        k = k.view(B, T, n_head, hs).transpose(1, 2)
        q = q.view(B, T, n_head, hs).transpose(1, 2)
        v = v.view(B, T, n_head, hs).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=dropout if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
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
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight
        self.apply(self._init_weights)
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
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


model = GPTLanguageModel()
m = model.to(device)
print(f"device: {device}  flash: {model.blocki[0].attn.flash}")
emb = vocab_size * n_embd
print(f"{sum(p.numel() for p in m.parameters())/1e6:.4f} M parameters "
      f"({emb/sum(p.numel() for p in m.parameters())*100:.0f}% of them in the tied embedding)")

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
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    sync()
    step_times.append(time.time() - t0)

steady = step_times[10:] or step_times
ms = sum(steady) / len(steady) * 1000
print(f"\ntotal wall time: {time.time()-t_start:.1f}s")
print(f"mean step: {ms:.2f} ms   throughput: {batch_size*block_size/(ms/1000):,.0f} tokens/s")

# seed generation with a real Khmer character rather than index 0, which is a newline
start = torch.tensor([[stoi.get('ក', 0)]], dtype=torch.long, device=device)
print("\n--- sample (· = zero-width space, ? = UNK) ---")
print(show(decode(m.generate(start, max_new_tokens=500)[0].tolist())))
