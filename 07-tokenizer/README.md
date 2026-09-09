# 07 · Let's build the GPT Tokenizer

- **Video:** *Let's build the GPT Tokenizer*
- **Reference:** [karpathy/minbpe](https://github.com/karpathy/minbpe)

## What it covers
Tokenization — the step before the model:
- why we don't feed raw characters or bytes to LLMs
- **Byte-Pair Encoding (BPE):** train a tokenizer that iteratively merges the most
  frequent byte pairs, then encode/decode text with it
- UTF-8 bytes, special tokens, regex splitting, and the quirks tokenizers cause

## Files
- `minbpe/` — Karpathy's BPE, with three bugs fixed (see the study, §8)
- `make_khmer_only.py` — build a Khmer-script-only corpus from a scraped one
- `bench_patterns2.py` · `bench_sp2.py` · `bench_sota.py` — the benchmarks

## Khmer tokenizer study
[**KHMER_TOKENIZER_STUDY.md**](KHMER_TOKENIZER_STUDY.md) — an empirical study of how
current tokenizers handle Khmer. A 4,000-token Khmer-only tokenizer beats every
production multilingual tokenizer tested; none allocates more than 0.79% of its
vocabulary to Khmer, and GPT-4's holds exactly one such token.
