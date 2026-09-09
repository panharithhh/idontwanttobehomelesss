"""
khmer_bpe_gpt.py - the same GPT, trained twice: once on characters, once on BPE tokens.

Architecture, steps and corpus are identical. The ONLY variable is the tokenizer, so
any difference in the samples is attributable to tokenization.

The comparison metric is BITS PER CHARACTER, not loss. Cross-entropy over an 87-way
char vocab and over a 2048-way BPE vocab are different scales and cannot be compared
directly. Normalising by characters makes them comparable:

    bpc = loss_nats * (tokens / characters) / ln(2)
"""
import math, os, time, tempfile
import torch, torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm

# ---- shared config: identical for both runs ----
batch_size, block_size = 32, 64
max_iters, eval_interval, eval_iters = 4000, 500, 100
learning_rate, n_embd, n_head, n_layer, dropout = 3e-4, 128, 4, 4, 0.1
BPE_VOCAB = 2048
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(1337)

text = open('khmer_input.txt', encoding='utf-8').read()
ZWSP = '​'
show = lambda s: s.replace(ZWSP, '·')

# ---- tokenizer A: characters ----
chars = sorted(set(text))
c_stoi = {c: i for i, c in enumerate(chars)}
c_itos = {i: c for c, i in c_stoi.items()}
char_ids = [c_stoi[c] for c in text]
char_dec = lambda l: ''.join(c_itos[i] for i in l)

# ---- tokenizer B: SentencePiece BPE, the winner from bench_sota.py ----
tmp = tempfile.mkdtemp()
f = os.path.join(tmp, 'km.txt'); open(f, 'w', encoding='utf-8').write(text)
spm.SentencePieceTrainer.train(
    input=f, model_prefix='khmer_bpe', vocab_size=BPE_VOCAB, model_type='bpe',
    character_coverage=1.0, split_by_whitespace=False,
    normalization_rule_name='identity', remove_extra_whitespaces=False,
    add_dummy_prefix=False, byte_fallback=True, minloglevel=2)
sp = spm.SentencePieceProcessor(model_file='khmer_bpe.model')
bpe_ids = sp.encode(text)

RUNS = [
    ("char-level", len(chars), char_ids, char_dec),
    (f"SP-BPE {BPE_VOCAB}", BPE_VOCAB, bpe_ids, lambda l: sp.decode(l)),
]
print(f"corpus: {len(text):,} chars")
print(f"  char-level -> {len(char_ids):,} tokens, vocab {len(chars)}")
print(f"  SP-BPE     -> {len(bpe_ids):,} tokens, vocab {BPE_VOCAB} "
      f"({len(text)/len(bpe_ids):.2f} chars/token)")
print(f"  block_size={block_size} tokens spans {block_size} chars vs "
      f"{block_size*len(text)/len(bpe_ids):.0f} chars\n")


class Block(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.c_attn = nn.Linear(n_embd, 3*n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4*n_embd, bias=False), nn.GELU(),
                                 nn.Linear(4*n_embd, n_embd, bias=False), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def attn(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        hs = C // n_head
        q, k, v = [t.view(B, T, n_head, hs).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout if self.training else 0.0,
                                           is_causal=True)
        return self.drop(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.tok = nn.Embedding(V, n_embd); self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(V) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd); self.head = nn.Linear(n_embd, V, bias=False)
        self.tok.weight = self.head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    @torch.no_grad()
    def generate(self, idx, n, temp=0.8):
        for _ in range(n):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


results = []
for name, V, ids, dec in RUNS:
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9 * len(data)); tr, va = data[:n], data[n:]
    tok_per_char = len(ids) / len(text)

    def batch(split):
        d = tr if split == 'train' else va
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    model = GPT(V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"=== {name} === {sum(p.numel() for p in model.parameters())/1e6:.3f}M params")
    t0 = time.time()
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            model.eval()
            with torch.no_grad():
                vl = torch.tensor([model(*batch('val'))[1].item() for _ in range(20)]).mean()
            model.train()
            print(f"  step {it:>5}: val loss {vl:.4f}  ({vl*tok_per_char/math.log(2):.4f} bits/char)")
        x, y = batch('train')
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        vl = torch.tensor([model(*batch('val'))[1].item() for _ in range(50)]).mean().item()
    bpc = vl * tok_per_char / math.log(2)
    start = torch.tensor([[ids[0]]], dtype=torch.long, device=device)
    sample = dec(model.generate(start, 200)[0].tolist())
    results.append((name, V, vl, bpc, time.time()-t0, sample))
    print(f"  final: val {vl:.4f}, {bpc:.4f} bits/char, {time.time()-t0:.0f}s\n")

print("=" * 78)
print(f"{'tokenizer':<16}{'vocab':>8}{'val loss':>10}{'bits/char':>12}{'train s':>9}")
print("-" * 78)
for name, V, vl, bpc, dt, _ in results:
    print(f"{name:<16}{V:>8,}{vl:>10.4f}{bpc:>12.4f}{dt:>9.0f}")
print("\nbits/char is the comparable metric. Lower is better.")
for name, V, vl, bpc, dt, sample in results:
    print(f"\n{'='*78}\n--- {name} SAMPLE ---\n{show(sample)}")
