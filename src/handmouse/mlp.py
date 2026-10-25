"""Rung 2: multilayer perceptron from scratch in NumPy (ReLU, softmax cross-entropy, backprop)."""
from itertools import pairwise

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in pairwise(sizes)]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X: np.ndarray) -> list[np.ndarray]:
        """Activations per layer; the last entry is the logits."""
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return acts

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def loss_grads(self, X: np.ndarray, y: np.ndarray, l2: float = 0.0):
        acts = self.forward(X)
        p = self.softmax(acts[-1])
        n = len(X)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean() + 0.5 * l2 * sum((W**2).sum() for W in self.W)
        d = p.copy()
        d[np.arange(n), y] -= 1
        d /= n
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d + l2 * self.W[i]
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)  # acts[i] is post-ReLU
        return loss, gW, gb

    def fit(self, X, y, epochs=200, lr=0.05, batch=64, l2=1e-4, momentum=0.9, seed=0):
        rng = np.random.default_rng(seed)
        vW = [np.zeros_like(W) for W in self.W]
        vb = [np.zeros_like(b) for b in self.b]
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                j = idx[s : s + batch]
                _, gW, gb = self.loss_grads(X[j], y[j], l2)
                for k in range(len(self.W)):
                    vW[k] = momentum * vW[k] - lr * gW[k]
                    vb[k] = momentum * vb[k] - lr * gb[k]
                    self.W[k] += vW[k]
                    self.b[k] += vb[k]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[-1].argmax(axis=1)


def gradient_check(model: MLP, X, y, l2: float = 1e-3, eps: float = 1e-5, n: int = 30,
                   seed: int = 0) -> float:
    """Max relative error between backprop and central-difference gradients on n random params."""
    _, gW, gb = model.loss_grads(X, y, l2)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for params, grads in ((model.W, gW), (model.b, gb)):
        for _ in range(n):
            k = rng.integers(len(params))
            pos = tuple(rng.integers(s) for s in params[k].shape)
            old = params[k][pos]
            params[k][pos] = old + eps
            up = model.loss_grads(X, y, l2)[0]
            params[k][pos] = old - eps
            down = model.loss_grads(X, y, l2)[0]
            params[k][pos] = old
            num, ana = (up - down) / (2 * eps), grads[k][pos]
            worst = max(worst, abs(num - ana) / max(abs(num) + abs(ana), 1e-8))
    return worst
