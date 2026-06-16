# 02 · makemore — MLP

- **Video:** *Building makemore Part 2: MLP*
- **Reference:** [karpathy/makemore](https://github.com/karpathy/makemore) · paper: Bengio et al. 2003

## What it covers
A multi-layer-perceptron character language model following Bengio et al. (2003):
- a context window of `block_size` previous characters
- an embedding table `C` mapping characters into a low-dim space
- a hidden `tanh` layer + output layer trained with `F.cross_entropy`
- train/dev/test splits, mini-batching, and tuning the learning rate

## Files
- `makemore_yay2.ipynb` — the part-2 working notebook
- `names.txt` — symlink to `../data/names.txt`
