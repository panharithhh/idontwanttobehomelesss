# 08 · Let's reproduce GPT-2 (124M)

- **Video:** *Let's reproduce GPT-2 (124M)*
- **Reference:** [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) · [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

## What it covers
The capstone: reproduce the 124M-parameter **GPT-2** from scratch.
- faithfully re-implement the GPT-2 architecture and load OpenAI's weights to validate
- train on a large dataset (FineWeb-Edu) with a real LLM training pipeline
- speedups: mixed precision, `torch.compile`, flash attention, gradient accumulation,
  distributed data parallel (DDP), LR schedule and weight decay
- evaluate against the GPT-2 baseline (e.g. HellaSwag)

## Data
Uses a large web-text dataset (downloaded/preprocessed), **not** `names.txt`.

## Files
- *(add your GPT-2 reproduction code here)*
