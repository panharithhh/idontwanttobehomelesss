# minbpe — Learning Notes

Notes for building the `BasicTokenizer` (`train` / `encode` / `decode`) in `minbpe.py`.

---

## `tokens = text.encode('utf-8')` — is `tokens` a list?

**No.** `.encode('utf-8')` returns a **`bytes`** object, not a list.

- `bytes` is like an *immutable* sequence of integers, each in the range **0–255**.
- It iterates like a list (you get ints out), which is why it's easy to confuse.

```python
text = "héllo"
tokens = text.encode('utf-8')

tokens          # b'h\xc3\xa9llo'   (é becomes TWO bytes)
list(tokens)    # [104, 195, 169, 108, 108, 111]   <- now it's a real list
tokens[0]       # 104  (an int)
len(tokens)     # 6
```

| | `bytes` | `list` |
|---|---|---|
| Mutable? | No (`tokens[0] = 5` fails) | Yes |
| Holds | ints 0–255 only | anything |
| Iterating gives | ints | the elements |

**Why it matters here:** BPE needs to *modify* the sequence while merging, so the first
step is almost always to convert it: turn the `bytes` into a `list` of ints first.

---

## What do `ids` and `idx` mean?

- **`ids`** — plural, "IDs". The **whole current sequence of tokens** (a list of ints).
  Starts as the raw UTF-8 byte values (0–255); as you merge, entries get replaced by
  new higher-numbered IDs.

- **`idx`** — singular, "index". A **single new token ID** assigned to a freshly merged
  pair. The first 256 IDs are taken by raw bytes, so new merges get `256, 257, 258, …` —
  each one is an `idx`.

Mental model during training:
1. Find the most common **pair** (two adjacent IDs) in `ids`.
2. Give that pair a new `idx` (e.g. `256`).
3. Replace every occurrence of the pair in `ids` with that `idx`.

You'll see them together in the `merges` dict: it maps `(id_a, id_b) -> idx`
("this pair of IDs becomes this new single ID").

Summary: `ids` = many tokens (the sequence), `idx` = one token (the new id).

---

## A reusable "teach me" prompt for Claude

Paste one of these when you want help *learning* instead of just getting the answer:

> **Be my tutor for this. Explain the concept and give hints, but don't write the code for me — let me try first, then review what I wrote and point out what's wrong and why.**

Shorter version:

> **Teach, don't solve: explain the idea, let me attempt the code, then check my work.**

Tips that make these work well with me:
- Say your **goal + level**: "I'm learning BPE from scratch, no copy-paste answers."
- Ask for **one step at a time** so you're not handed the whole solution.
- When stuck, ask for a **hint or a leading question**, not the code.
- After you write something, paste it and ask me to **review and explain mistakes**.
