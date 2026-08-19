import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
# NOTE: scaled down from Karpathy's actual numbers (batch_size=64, block_size=256,
# n_embd=384, n_head=6, n_layer=6, dropout=0.2) so this trains in reasonable time
# on CPU. Bump these up if you've got a GPU.
batch_size = 32
block_size = 32
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
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


class BigramLanguageModel(nn.Module):
    """ the video's first checkpoint - no attention yet, just embeddings + lm_head """

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        

    def forward(self, idx, targets=None):

        
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)  
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        else :
            logits = logits.view(B*T, vocab_size) # B , T 
            targets = targets.view(B*T)   
            loss = F.cross_entropy(logits, target=targets) # 
            return logits, loss

    def generate(self, idx, max_new_tokens):
        # TODO: same sampling loop as idk.py -
        for token in range(max_new_tokens):
            logits, loss = self.forward(idx) # B , T ,C 
            logits = logits[:,-1,:]  # B , C 
            # last positin of the output
            probs = torch.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
            
        


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) 
        self.query= nn.Linear(n_embd, head_size, bias=False)
        self.value= nn.Linear(n_embd, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B,T,C = x.shape 
        # TODO: k = self.key(x)      -> (B, T, head_size)
        k = self.key(x) # B , T, C
        q = self.query(x) # B , T,C' 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  
        #(T,C) (C,T) = (T,T) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
 
        wei = F.softmax(wei, dim = -1)  
        
        wei = self.dropout(wei) 
        # to prevent from it memoriezing from learning so we will just fill some of the row with zero's
    
        v = self.value(x)
        out = wei @ v
        return  out


class GPTLanguageModel(nn.Module):
    """ checkpoint 2: BigramLanguageModel + one head of self-attention plugged directly in """

    def __init__(self):
        super().__init__()
        # TODO: self.token_embedding_table = nn.Embedding(vocab_size, n_embd)     - same as Bigram
        # TODO: self.position_embedding_table = nn.Embedding(block_size, n_embd)  - same as Bigram
        # TODO: self.sa_head = Head(n_embd)   # NEW: one head of self-attention, head_size == n_embd
        # TODO: self.lm_head = nn.Linear(n_embd, vocab_size)                      - same as Bigram
        raise NotImplementedError

    def forward(self, idx, targets=None):
        # TODO: B, T = idx.shape
        # TODO: tok_emb = self.token_embedding_table(idx)
        # TODO: pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # TODO: x = tok_emb + pos_emb          - identical to BigramLanguageModel, up to here
        # TODO: x = self.sa_head(x)            - NEW: x now gets enriched with context via attention
        # TODO: logits = self.lm_head(x)       - identical to BigramLanguageModel, from here on
        #
        # then the exact same loss computation as BigramLanguageModel.forward:
        #   if targets is None: return logits, None
        #   else: reshape logits to (B*T, vocab_size), targets to (B*T,), F.cross_entropy, return loss, logits
        raise NotImplementedError

    def generate(self, idx, max_new_tokens):
        # same sampling loop as BigramLanguageModel.generate, with ONE addition:
        # TODO: before calling self(...), crop idx down to the last block_size tokens:
        #       idx_cond = idx[:, -block_size:]
        #       (why is this needed here but not in BigramLanguageModel? think about what
        #        position_embedding_table's row count limits you to, once idx grows past block_size)
        raise NotImplementedError


# --- checkpoint 1: train the bigram model (no attention yet) ---
# once GPTLanguageModel is implemented above, this
# `model = BigramLanguageModel()` line gets swapped for `model = GPTLanguageModel()`
model = BigramLanguageModel()
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
print(decode(m.generate(context, max_new_tokens=20)[0].tolist()))
