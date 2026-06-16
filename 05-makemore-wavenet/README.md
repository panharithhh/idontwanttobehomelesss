# 05 · makemore — Building a WaveNet

- **Video:** *Building makemore Part 5: Building a WaveNet*
- **Reference:** [karpathy/makemore](https://github.com/karpathy/makemore) · paper: WaveNet (van den Oord et al. 2016)

## What it covers
Make the network deeper and more structured:
- grow the flat MLP into a hierarchical, tree-like **WaveNet**-style architecture that
  fuses characters two at a time across layers
- build a small PyTorch-like module API (`Linear`, `BatchNorm1d`, `Tanh`, `Embedding`,
  `Flatten`, `Sequential`) — `torch.nn`-style containers, by hand
- a more realistic train/eval loop and reasoning about tensor shapes

## Files
- `names.txt` — symlink to `../data/names.txt`
- *(add your part-5 notebook / script here)*
