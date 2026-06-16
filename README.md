# Neural Networks: Zero to Hero

My work through Andrej Karpathy's [**Neural Networks: Zero to Hero**](https://karpathy.ai/zero-to-hero.html) course,
building everything from scratch — from a tiny autograd engine up to a reproduction of GPT-2.

Each part lives in its own folder **and** its own git branch (so I can work on one part at a time
without the others getting in the way).

## Course map

| # | Folder | Topic | Reference repo |
|---|--------|-------|----------------|
| 00 | [`00-micrograd/`](00-micrograd/) → **own repo** | Backprop & a scalar autograd engine | [panharithhh/micrograd](https://github.com/panharithhh/micrograd) · [karpathy/micrograd](https://github.com/karpathy/micrograd) |
| 01 | [`01-makemore-bigram/`](01-makemore-bigram/) | Bigram counts + a 1-layer neural net | [karpathy/makemore](https://github.com/karpathy/makemore) |
| 02 | [`02-makemore-mlp/`](02-makemore-mlp/) | MLP language model (Bengio 2003) | [karpathy/makemore](https://github.com/karpathy/makemore) |
| 03 | [`03-makemore-batchnorm/`](03-makemore-batchnorm/) | Activations, gradients & BatchNorm | [karpathy/makemore](https://github.com/karpathy/makemore) |
| 04 | [`04-makemore-backprop-ninja/`](04-makemore-backprop-ninja/) | Manual backprop ("backprop ninja") | [karpathy/makemore](https://github.com/karpathy/makemore) |
| 05 | [`05-makemore-wavenet/`](05-makemore-wavenet/) | Deeper, WaveNet-style hierarchy | [karpathy/makemore](https://github.com/karpathy/makemore) |
| 06 | [`06-gpt/`](06-gpt/) | Build GPT from scratch (a Transformer) | [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) |
| 07 | [`07-tokenizer/`](07-tokenizer/) | The GPT tokenizer (byte-pair encoding) | [karpathy/minbpe](https://github.com/karpathy/minbpe) |
| 08 | [`08-gpt2/`](08-gpt2/) | Reproduce GPT-2 (124M) | [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) |

## Branches

One branch per part that lives in this repo, named to match its folder:

```
01-makemore-bigram
02-makemore-mlp
03-makemore-batchnorm
04-makemore-backprop-ninja
05-makemore-wavenet
06-gpt
07-tokenizer
08-gpt2
```

Switch to the part you're working on, e.g. `git checkout 03-makemore-batchnorm`.

> Part **00 (micrograd)** has no branch here — it lives in its own repo
> ([panharithhh/micrograd](https://github.com/panharithhh/micrograd), a sibling folder at `../micrograd`).
> [`00-micrograd/`](00-micrograd/) is just a pointer to it.

## Dataset

The shared names dataset lives in [`data/names.txt`](data/names.txt). Every makemore part folder has a
`names.txt` symlink pointing back to it, so existing scripts that do `open('names.txt')` keep working.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch matplotlib numpy jupyter
```
