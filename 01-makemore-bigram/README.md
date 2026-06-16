# 01 · makemore — bigram

- **Video:** *The spelled-out intro to language modeling: building makemore*
- **Reference:** [karpathy/makemore](https://github.com/karpathy/makemore)

## What it covers
Character-level language modeling on names, two ways:
1. **Counts (bigram):** build a 27×27 matrix `N` of character-pair counts, normalize into
   probabilities `P`, sample new names, and evaluate with negative log-likelihood.
2. **Neural net:** the same bigram model as a single linear layer trained with gradient
   descent + `F.cross_entropy` — showing the count table and the trained weights converge.

## Files
- `bigram_counts.py` — bigram via the counts matrix + log-likelihood (was `2`)
- `makemorefromscratch_1.py` — bigram built up from scratch
- `makemore.ipynb` — the full part-1 notebook (counts → `plt.imshow` → neural-net version)
- `names.txt` — symlink to `../data/names.txt`

## Run
```bash
python bigram_counts.py
```
