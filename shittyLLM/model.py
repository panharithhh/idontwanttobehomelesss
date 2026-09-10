import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, n_embd, head_size, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        
        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)  # (B, T, hs)
        out = wei @ v      # (B, T, hs)
        return out


class MultiheadedAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    """ Simple linear projection followed by non-linearity """
    def __init__(self, n_embd, dropout=0.2):
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
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadedAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """ Complete Decoder-only GPT Transformer """
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.2, device='cpu'):
        super().__init__()
        self.block_size = block_size
        self.device = device
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                           # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, C)
        x = tok_emb + pos_emb                                               # (B, T, C)
        x = self.blocks(x)                                                  # (B, T, C)
        x = self.ln_f(x)                                                    # (B, T, C)
        logits = self.lm_head(x)                                            # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop context to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]             # focus on the last time step
            probs = F.softmax(logits, dim=-1)     # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # sample next token
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
