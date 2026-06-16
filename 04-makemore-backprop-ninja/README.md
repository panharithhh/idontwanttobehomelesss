# 04 · makemore — Becoming a Backprop Ninja

- **Video:** *Building makemore Part 4: Becoming a Backprop Ninja*
- **Reference:** [karpathy/makemore](https://github.com/karpathy/makemore)

## What it covers
Backpropagate **by hand** through the entire MLP + BatchNorm network from part 3 —
no `loss.backward()`. You manually derive and code the gradient for every intermediate
tensor (cross-entropy, the BatchNorm layer, the `tanh`, the matmuls, the embedding) and
check each one against PyTorch's autograd. The point: stop treating backprop as magic.

## Files
- `names.txt` — symlink to `../data/names.txt`
- *(add your part-4 notebook / script here)*
