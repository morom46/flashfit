"""
model.py - Step 2: train the full-size ("float") fault classifier.

A small MLP (Multi-Layer Perceptron: layers of weighted sums + ReLU) is trained
with scikit-learn. We then pull the weights out as plain NumPy arrays, because
the optimisers need to squash those weights thousands of times and re-check
accuracy quickly.

Network shape: 40 inputs -> 64 -> 48 -> 32 -> 6 classes   (4 weight layers)

Run:  python model.py      (needs dataset.npz from data.py)
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

HIDDEN = (64, 48, 32)


def forward(weights, biases, X):
    """Run the network by hand: (X @ W + b), ReLU, repeat. Returns class numbers."""
    a = X
    for i, (W, b) in enumerate(zip(weights, biases)):
        a = a @ W + b
        if i < len(weights) - 1:              # ReLU on hidden layers only
            a = np.maximum(a, 0)
    return np.argmax(a, axis=1)


def accuracy(weights, biases, X, y):
    """Fraction of recordings classified correctly (0.0 to 1.0)."""
    return float(np.mean(forward(weights, biases, X) == y))


def train(data, seed=0):
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, activation="relu",
                        max_iter=400, alpha=1e-2, random_state=seed)
    clf.fit(data["X_train"], data["y_train"])
    return clf.coefs_, clf.intercepts_


def load_model(path="model_float.npz"):
    f = np.load(path)
    n_layers = len(f.files) // 2
    weights = [f[f"W{i}"] for i in range(n_layers)]
    biases = [f[f"b{i}"] for i in range(n_layers)]
    return weights, biases


if __name__ == "__main__":
    data = np.load("dataset.npz")
    weights, biases = train(data)
    np.savez("model_float.npz",
             **{f"W{i}": W for i, W in enumerate(weights)},
             **{f"b{i}": b for i, b in enumerate(biases)})

    n_weights = sum(W.size for W in weights)
    n_biases = sum(b.size for b in biases)
    print(f"Weights: {n_weights}, biases: {n_biases}")
    print(f"Float32 size: {(n_weights + n_biases) * 4 / 1024:.1f} KB")
    for part in ("train", "val", "test"):
        acc = accuracy(weights, biases, data["X_" + part], data["y_" + part])
        print(f"{part:5s} accuracy: {acc * 100:.2f}%")
    print("Saved model_float.npz")
