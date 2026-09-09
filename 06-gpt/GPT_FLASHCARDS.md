# Andrej Karpathy: "Let's Build GPT from Scratch" — Master Flashcards & Study Guide

Designed for quick review, active recall, Anki import, and NotebookLM ingestion.

---

## 1. High-Level Architecture & Concepts

### Card 1.1: What is the high-level objective of a Generative Pretrained Transformer (GPT)?
- **Front (Question):** What is the core training objective of a Decoder-only Transformer (like GPT)?
- **Back (Answer):** Next-token prediction (autoregressive language modeling). Given a sequence of tokens $x_1, x_2, \dots, x_t$, predict the probability distribution over the vocabulary for the next token $x_{t+1}$ using cross-entropy loss.

---

### Card 1.2: Communication vs. Computation
- **Front (Question):** In a Transformer block, what is the conceptual difference in roles between the Self-Attention layer and the Feed-Forward layer?
- **Back (Answer):**
  - **Self-Attention = Communication:** Tokens look at other tokens and aggregate context across the sequence.
  - **Feed-Forward = Computation:** Each token individually processes and computes on the information it just gathered, with no cross-token interaction.

---

## 2. Embeddings & Inputs

### Card 2.1: Token vs. Position Embeddings
- **Front (Question):** Why do Transformers need Positional Embeddings in addition to Token Embeddings?
- **Back (Answer):** Self-attention is inherently a **set operation** (permutation invariant). Without positional embeddings, shuffling the order of words in a sentence would produce the exact same attention outputs. Positional embeddings inject information about where each token is located in the sequence.

---

### Card 2.2: Embedding Combination
- **Front (Question):** How are token embeddings and positional embeddings combined in GPT?
- **Back (Answer):** They are element-wise **added together**:
  `x = tok_emb + pos_emb` where both tensors have shape `(B, T, n_embd)`.

---

## 3. Attention Mechanism (Self-Attention Head)

### Card 3.1: Query, Key, and Value Intuition
- **Front (Question):** What do the Query ($Q$), Key ($K$), and Value ($V$) vectors represent conceptually?
- **Back (Answer):**
  - **Query ($Q$):** *"What am I (the current token) looking for?"*
  - **Key ($K$):** *"What information do I (a source token) contain?"*
  - **Value ($V$):** *"If you find me relevant, here is the actual information I will give you."*

---

### Card 3.2: Attention Weights Formula & Computation
- **Front (Question):** How are attention weights calculated from Queries and Keys?
- **Back (Answer):**
  $$\text{wei} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{mask}\right)$$
  - Dot product ($Q K^T$) measures alignment / affinity between tokens.
  - Scaled by $\frac{1}{\sqrt{d_k}}$ to keep variance around 1 and avoid vanishing gradients in softmax.
  - Masked with $-\infty$ for future tokens.
  - `softmax` turns scores into valid probability distributions (summing to 1 per row).

---

### Card 3.3: Why Scaled Dot-Product?
- **Front (Question):** Why do we divide the dot product $Q K^T$ by $\sqrt{\text{head\_size}}$?
- **Back (Answer):** In high dimensions, the dot product between two unit-variance vectors has variance equal to $\text{head\_size}$. Dividing by $\sqrt{\text{head\_size}}$ normalizes the variance back to 1.0, preventing `softmax` from becoming too peaky / saturated (which causes zero gradients).

---

### Card 3.4: Causal Masking (The Lower Triangular Mask)
- **Front (Question):** What is the purpose of `tril` (lower triangular matrix) and `.masked_fill(tril == 0, float('-inf'))`?
- **Back (Answer):** It enforces the **causal (autoregressive) constraint**: tokens are only allowed to communicate with past and present tokens ($t' \le t$). Setting future positions to $-\infty$ forces `softmax` to give them a weight of `0.0`.

---

### Card 3.5: Self-Attention vs. Cross-Attention
- **Front (Question):** What makes an attention layer **"Self-Attention"** vs **"Cross-Attention"**?
- **Back (Answer):**
  - **Self-Attention:** $Q$, $K$, and $V$ all come from the same source sequence ($x$).
  - **Cross-Attention:** $Q$ comes from the current decoder sequence, but $K$ and $V$ come from an external source (e.g. an encoder's output in translation models).

---

## 4. Multi-Head Attention & Transformer Blocks

### Card 4.1: Multi-Head Attention Motivation
- **Front (Question):** Why use Multi-Head Attention instead of one single large Attention Head?
- **Back (Answer):** Multiple heads allow the model to jointly attend to information from **different representation subspaces** simultaneously (e.g., one head tracks grammatical subject-verb pairs, another tracks pronoun antecedents, another tracks punctuation).

---

### Card 4.2: Multi-Head Implementation
- **Front (Question):** If embedding dimension is $C=64$ and number of heads is $h=4$, what is the dimension of each head, and how are outputs combined?
- **Back (Answer):**
  - $\text{head\_size} = C // h = 64 // 4 = 16$.
  - Each head outputs `(B, T, 16)`.
  - All 4 heads are concatenated along the channel dimension back to `(B, T, 64)`, followed by a linear projection.

---

### Card 4.3: Residual (Skip) Connections
- **Front (Question):** Why are residual connections ($x = x + \text{sublayer}(x)$) crucial in deep Transformers?
- **Back (Answer):** They provide a direct highway for gradients to flow uninterrupted during backpropagation from the loss all the way back to the initial embeddings, enabling training of very deep networks without vanishing/exploding gradients.

---

### Card 4.4: LayerNorm vs. BatchNorm
- **Front (Question):** How does Layer Normalization differ from Batch Normalization in Transformers?
- **Back (Answer):**
  - **BatchNorm:** Normalizes across the batch dimension for each feature (problematic for variable sequence lengths and small batch sizes).
  - **LayerNorm:** Normalizes across the feature/channel dimension independently for each token in each batch element (no cross-batch dependencies).

---

### Card 4.5: Pre-LN vs. Post-LN
- **Front (Question):** What is the modern standard placement of LayerNorm in GPT (Pre-LN)?
- **Back (Answer):**
  - **Pre-LN:** $x = x + \text{Attention}(\text{LayerNorm}(x))$
  - **Post-LN (Original 2017 Transformer):** $x = \text{LayerNorm}(x + \text{Attention}(x))$
  - Pre-LN is preferred because the residual stream remains clean, making training significantly more stable.

---

## 5. Output, Loss, and Generation

### Card 5.1: Logits vs. Probabilities vs. Loss
- **Front (Question):** What are `logits`, how are they converted into probabilities, and how is loss calculated?
- **Back (Answer):**
  - **Logits:** Raw, unnormalized score outputs from `lm_head(x)` of shape `(B, T, vocab_size)`.
  - **Probabilities:** `softmax(logits, dim=-1)`.
  - **Loss:** `F.cross_entropy(logits, targets)`, which computes negative log-likelihood over the predicted distribution against ground truth next tokens.

---

### Card 5.2: Autoregressive Sampling Loop
- **Front (Question):** Describe the steps in `GPTLanguageModel.generate(idx, max_new_tokens)`.
- **Back (Answer):**
  1. Crop context to the last `block_size` tokens: `idx_cond = idx[:, -block_size:]`
  2. Run forward pass to get logits: `logits, _ = self(idx_cond)`
  3. Pluck the logits at the very last time step: `logits = logits[:, -1, :]`
  4. Apply softmax to get probabilities: `probs = F.softmax(logits, dim=-1)`
  5. Sample next token index: `idx_next = torch.multinomial(probs, num_samples=1)`
  6. Append `idx_next` to `idx` and repeat.

---

## 6. Tensor Shape Reference Cheat Sheet

| Step | Tensor / Variable | Shape |
| :--- | :--- | :--- |
| **Input Batch** | `idx` | `(B, T)` |
| **Embeddings** | `tok_emb + pos_emb` | `(B, T, C)` |
| **Key / Query / Value** | `k, q, v` | `(B, T, head_size)` |
| **Attention Weights** | `wei = q @ k.T` | `(B, T, T)` |
| **Head Output** | `wei @ v` | `(B, T, head_size)` |
| **Multi-Head Output** | `torch.cat(...)` | `(B, T, C)` |
| **FeedForward Output** | `ffwd(x)` | `(B, T, C)` |
| **Logits** | `lm_head(x)` | `(B, T, vocab_size)` |
