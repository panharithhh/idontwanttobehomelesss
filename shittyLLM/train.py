import torch
from tokenizer import BasicTokenizer
from model import GPTLanguageModel

# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------
batch_size = 64
block_size = 256
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 100
n_embd = 128      # lightweight & fast for local testing
n_head = 4
n_layer = 4
dropout = 0.2
vocab_size = 300  # 256 base bytes + 44 BPE merges

torch.manual_seed(1337)
print(f"Using device: {device}")

# --------------------------------------------------
# 1. Load Data & Train Tokenizer
# --------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# For training the tokenizer fast, take a slice of text (or the full text)
print("Training BasicTokenizer (BPE)...")
tokenizer = BasicTokenizer()
tokenizer.train(text[:50000], vocab_size=vocab_size, verbose=False)
print(f"Tokenizer trained with vocab_size={vocab_size}")

# --------------------------------------------------
# 2. Encode Dataset
# --------------------------------------------------
print("Encoding dataset into tokens...")
encoded_ids = tokenizer.encode(text)
data = torch.tensor(encoded_ids, dtype=torch.long)
print(f"Total tokens: {len(data)} (from {len(text)} characters)")

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
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

# --------------------------------------------------
# 3. Initialize Model & Optimizer
# --------------------------------------------------
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
    device=device
).to(device)

print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --------------------------------------------------
# 4. Training Loop
# --------------------------------------------------
print("Starting training...")
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --------------------------------------------------
# 5. Generate Sample Text
# --------------------------------------------------
print("\n--- GENERATED TEXT ---")
start_context = "ROMEO:"
start_ids = tokenizer.encode(start_context)
context = torch.tensor([start_ids], dtype=torch.long, device=device)

generated_ids = model.generate(context, max_new_tokens=200)[0].tolist()
print(tokenizer.decode(generated_ids))
