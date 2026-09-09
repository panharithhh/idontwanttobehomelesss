import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
# Karpathy's full config from the video: ~10.8M parameters.
# Measured on this machine: roughly 30-40 min for 5000 steps on mps.
batch_size = 64   # B
block_size = 256  # T
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------
# SHAPE LEGEND - every comment below uses these letters
#   B  = batch_size  = 64     how many sequences at once
#   T  = block_size  = 256    tokens of context (the "time" axis)
#   C  = n_embd      = 384    channels per token (the residual stream width)
#   hs = head_size   = 64     C // n_head, one head's slice of the channels
#   V  = vocab_size  = 65     unique characters in input.txt
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

data = torch.tensor(encode(text), dtype=torch.long)   # (1115394,) one int per char
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))   # (B,)  random start offsets
    x = torch.stack([data[i:i+block_size] for i in ix])         # (B, T) = (64, 256)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])     # (B, T) same window, shifted by 1
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

class MultiheadedAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__() 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # projection back into the residual stream, so the heads get mixed
        # together before the result is added onto x
        self.proj = nn.Linear(head_size * num_heads, n_embd)   # weight (384, 384)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):                                      # x: (B, T, C)
        # each head returns (B, T, hs); 6 of them concatenated on the channel axis
        out = torch.cat([h(x) for h in self.heads], dim=-1)     # (B, T, 6*64) = (B, T, 384) = (B, T, C)
        return self.dropout(self.proj(out))                    # (B, T, C) - shape-preserving, so it can feed a residual
        
        
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity - gives each token
        time to individually process what attention just gathered for it """

    def __init__(self, n_embd):
        super().__init__()
        
        self.net = nn.Sequential(
                   nn.Linear(n_embd, 4*n_embd),   # weight (1536, 384):  (B,T,384) -> (B,T,1536)
                   nn.ReLU(),                     #                      (B,T,1536) unchanged
                   nn.Linear(4*n_embd, n_embd),   # weight (384, 1536):  (B,T,1536) -> (B,T,384)
                   nn.Dropout(dropout),
               )

    def forward(self, x):                         # x: (B, T, C)
        return self.net(x)                        # (B, T, C) - out and back, same width

class Block(nn.Module):
    """ communication (attention) followed by computation (feedforward) """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head   # 384 // 6 = 64, so 6 heads concat back to exactly 384
        self.sa = MultiheadedAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        # one layernorm per sublayer. each normalizes a token's own n_embd numbers,
        # using nothing from other tokens or other batch elements
        self.ln1 = nn.LayerNorm(n_embd)   # weight (384,), bias (384,)
        self.ln2 = nn.LayerNorm(n_embd)   # normalizes each token's own 384 numbers

    def forward(self, x):                 # x: (B, T, C)
        # pre-norm: the norm sits INSIDE the branch, so the skip path stays a clean x
        x = x + self.sa(self.ln1(x))      # (B,T,C) + (B,T,C) -> (B,T,C), elementwise
        x = x + self.ffwd(self.ln2(x))    # (B,T,C) + (B,T,C) -> (B,T,C)
        return x                          # (B, T, C) in, (B, T, C) out - that is why Blocks stack


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)    # weight (64, 384): C -> hs
        self.query= nn.Linear(n_embd, head_size, bias=False)   # weight (64, 384)
        self.value= nn.Linear(n_embd, head_size, bias=False)   # weight (64, 384)
        
        # (T, T) lower-triangular of ones. a buffer, not a Parameter: it moves to the
        # GPU with the model and lands in state_dict, but the optimizer never sees it
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # (256, 256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B,T,C = x.shape                     # (64, 256, 384)
        k = self.key(x)                     # (B, T, hs) = (64, 256, 64)
        q = self.query(x)                   # (B, T, hs) = (64, 256, 64)

        # every query dotted with every key. this is the ONLY line in the whole file
        # where information moves between token positions
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        # (B,T,hs) @ (B,hs,T) -> (B,T,T) = (64, 256, 256)   one score per (query, key) pair
        # the * 64**-0.5 keeps the scores at variance ~1 so softmax does not saturate

        # tril[:T,:T] is (T,T) and broadcasts across the batch dim
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))   # (B, T, T) 
        wei = F.softmax(wei, dim = -1)      # (B, T, T)  each ROW sums to 1 (dim=-1 = over keys)        
        wei = self.dropout(wei) 
        # to prevent from it memoriezing from learning so we will just fill some of the row with zero's 
        v = self.value(x)                   # (B, T, hs) = (64, 256, 64)
        out = wei @ v                       # (B,T,T) @ (B,T,hs) -> (B,T,hs) = (64, 256, 64)
        return  out


class GPTLanguageModel(nn.Module):
    """ token/position embeddings -> n_layer Blocks -> final layernorm -> lm_head """

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)      # (V, C) = (65, 384)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)   # (T, C) = (256, 384)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)             # one last norm before the output head
        self.lm_head = nn.Linear(n_embd, vocab_size) # weight (65, 384): C -> V

    def forward(self, idx, targets=None):
        B, T = idx.shape                                                   # (64, 256) int64
        tok_emb = self.token_embedding_table(idx)                          # (B, T, C) = (64, 256, 384)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C) = (256, 384)
        x = tok_emb + pos_emb                # (B,T,C) + (T,C) broadcasts -> (B, T, C)
        x = self.blocks(x)                   # (B, T, C) - 6 Blocks, shape never changes
        x = self.ln_f(x)                     # (B, T, C)
        logits = self.lm_head(x)             # (B, T, V) = (64, 256, 65)
        # note: a prediction at EVERY position, not just the last one

        if targets is None:
            return logits, None
        else:
            # careful: this C is rebound to vocab_size (65), NOT n_embd (384)
            B, T, C = logits.shape                    # (64, 256, 65)
            logits = logits.view(B*T, C)              # (16384, 65)  cross_entropy wants 2D
            targets = targets.view(B*T)               # (16384,)     and 1D targets
            loss = F.cross_entropy(logits, targets)   # scalar, averaged over 16384 predictions
            return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx starts (B, 1) and grows by one column per step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]        # (B, <=256) crop: position table only has 256 rows
            logits, loss = self(idx_cond)          # (B, T_cur, V)
            logits = logits[:, -1, :]              # (B, V) - only the LAST position predicts the next token
            probs = F.softmax(logits, dim=-1)      # (B, V) probabilities over the 65 characters
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1) one sampled id per sequence
            idx = torch.cat((idx, idx_next), dim=1)              # (B, T_cur + 1)
        return idx

model = GPTLanguageModel()

m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

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

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
