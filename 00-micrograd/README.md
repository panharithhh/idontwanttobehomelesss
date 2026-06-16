# 00 · micrograd

> **This part lives in its own repository.** This folder is only a pointer.

- **Local:** `../micrograd` (sibling folder)
- **GitHub:** https://github.com/panharithhh/micrograd
- **Reference:** [karpathy/micrograd](https://github.com/karpathy/micrograd)
- **Video:** *The spelled-out intro to neural networks and backpropagation: building micrograd*

## What it covers
A tiny scalar-valued autograd engine (`Value`) that builds a computation graph and does
reverse-mode automatic differentiation (backpropagation), plus a small neural-net library
(`Neuron`, `Layer`, `MLP`) built on top of it. This is the foundation everything else in the
course rests on.
