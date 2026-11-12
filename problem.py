"""
problem.py - Step 3: the optimisation problem (chromosome + score card).

The question
------------
The float model is ~29 KB. The microcontroller only has FLASH_BUDGET_BYTES
spare. How hard should we squash each layer so the model fits AND keeps as
much accuracy as possible?

Two ways to squash a layer:
  * quantise - store each weight with fewer bits (8, 6, 5, 4, 3 or 2)
  * prune    - delete the smallest weights (0%, 10%, ... 80% of them)

Chromosome (the "recipe"): 8 integers
    [bits_L1, bits_L2, bits_L3, bits_L4, prune_L1, prune_L2, prune_L3, prune_L4]
    e.g. [4, 3, 4, 8, 20, 50, 30, 0]

Score card (LOWER = BETTER, same convention as a golf score):
    score = validation error in %                    (accuracy we lost)
          + overshoot penalty if the model is too big (constraint)
          + 0.5 * (size / budget)                    (tie-break: smaller wins)

Search space: 6^4 * 9^4 = 8,503,056 recipes.
"""

import numpy as np

BIT_CHOICES = [2, 3, 4, 5, 6, 8]
PRUNE_CHOICES = [0, 10, 20, 30, 40, 50, 60, 70, 80]     # percent removed
N_LAYERS = 4
N_GENES = 2 * N_LAYERS
GENE_SPACE = [BIT_CHOICES] * N_LAYERS + [PRUNE_CHOICES] * N_LAYERS

FLASH_BUDGET_BYTES = 3072        # 3 KB for weights; the float model is ~29 KB
OVERSHOOT_FIXED = 5.0            # flat penalty for breaking the budget at all
OVERSHOOT_SLOPE = 100.0          # plus this much per 100% over budget
SIZE_TIEBREAK = 0.5


def prune_weights(W, prune_pct):
    """Set the smallest prune_pct% of weights (by absolute size) to zero."""
    if prune_pct == 0:
        return W.copy()
    threshold = np.percentile(np.abs(W), prune_pct)
    return np.where(np.abs(W) >= threshold, W, 0.0)


def quantise_weights(W, bits):
    """Round weights onto a grid with 2^bits - 1 evenly spaced levels.

    Example with bits=2: every weight becomes -max, 0 or +max.
    Returns (integer codes, scale) where  real_weight ~= code * scale.
    """
    levels = 2 ** (bits - 1) - 1                 # e.g. 8 bits -> codes -127..127
    scale = np.max(np.abs(W)) / levels
    if scale == 0:                               # layer is all zeros
        return np.zeros_like(W, dtype=np.int16), 1.0
    codes = np.clip(np.round(W / scale), -levels, levels).astype(np.int16)
    return codes, float(scale)


def layer_size_bits(n_weights, n_biases, bits, prune_pct, n_kept):
    """Flash cost of one layer, in bits.

    kept weights * bits
    + a 1-bit-per-weight map saying which weights survived (only if pruned)
    + biases kept at full 32-bit precision (there are very few of them)

    Note the trap: light pruning at low bit-widths can make a layer BIGGER,
    because the map costs more than the deleted weights saved.
    """
    mask_bits = n_weights if prune_pct > 0 else 0
    return n_kept * bits + mask_bits + n_biases * 32


class FlashFitProblem:
    """Holds the float model + validation data and scores any chromosome."""

    def __init__(self, weights, biases, X_val, y_val,
                 budget_bytes=FLASH_BUDGET_BYTES):
        # float32 = the number format a microcontroller would use (and faster)
        self.weights = [W.astype(np.float32) for W in weights]
        self.biases = [b.astype(np.float32) for b in biases]
        self.X_val, self.y_val = X_val.astype(np.float32), y_val
        self.budget_bytes = budget_bytes
        self.n_evaluations = 0       # how many times an optimiser asked for a score
        self._cache = {}             # recipe -> score card (same recipe, same answer)
        self._layer1_out = {}        # (bits, prune) of layer 1 -> its output

        # Pre-compute every squashed version of every layer once (4 * 6 * 9 = 216),
        # so scoring a recipe is just "pick 4 ready-made layers and run the net".
        self._squashed = {}
        for layer, (W, b) in enumerate(zip(self.weights, self.biases)):
            for bits in BIT_CHOICES:
                for prune_pct in PRUNE_CHOICES:
                    pruned = prune_weights(W, prune_pct)
                    codes, scale = quantise_weights(pruned, bits)
                    n_kept = int(np.count_nonzero(pruned))
                    size = layer_size_bits(W.size, b.size, bits, prune_pct, n_kept)
                    self._squashed[(layer, bits, prune_pct)] = (
                        (codes * scale).astype(np.float32), size)

    # ---- helpers -----------------------------------------------------------
    def decode(self, chromosome):
        """Split the 8 genes into (bits per layer, prune % per layer)."""
        genes = [int(g) for g in chromosome]
        return genes[:N_LAYERS], genes[N_LAYERS:]

    def squashed_weights(self, chromosome):
        bits, prune = self.decode(chromosome)
        return [self._squashed[(i, bits[i], prune[i])][0] for i in range(N_LAYERS)]

    def size_bytes(self, chromosome):
        bits, prune = self.decode(chromosome)
        total_bits = sum(self._squashed[(i, bits[i], prune[i])][1]
                         for i in range(N_LAYERS))
        return int(np.ceil(total_bits / 8))

    def predict(self, chromosome, X=None):
        """Class predicted for each recording by the squashed model."""
        X = self.X_val if X is None else X.astype(np.float32)
        layers = self.squashed_weights(chromosome)
        on_val = X is self.X_val

        # Layer 1 only has 54 possible versions, so on the validation set we
        # remember its output instead of recomputing it (memoisation).
        key = (int(chromosome[0]), int(chromosome[N_LAYERS]))
        if on_val and key in self._layer1_out:
            a = self._layer1_out[key]
        else:
            a = np.maximum(X @ layers[0] + self.biases[0], 0)
            if on_val:
                self._layer1_out[key] = a

        for i in range(1, N_LAYERS):
            a = a @ layers[i] + self.biases[i]
            if i < N_LAYERS - 1:
                a = np.maximum(a, 0)              # ReLU on hidden layers only
        return np.argmax(a, axis=1)

    def error_pct(self, chromosome, X=None, y=None):
        """Percent of recordings the squashed model gets wrong."""
        y = self.y_val if y is None else y
        return float(100.0 * np.mean(self.predict(chromosome, X) != y))

    # ---- the score card ----------------------------------------------------
    def score_card(self, chromosome):
        """Full breakdown for one recipe (a dict). Lower 'score' is better."""
        key = tuple(int(g) for g in chromosome)
        if key in self._cache:
            return self._cache[key]

        size = self.size_bytes(key)
        error = self.error_pct(key)
        over = max(0, size - self.budget_bytes) / self.budget_bytes
        penalty = 0.0 if over == 0 else OVERSHOOT_FIXED + OVERSHOOT_SLOPE * over
        tiebreak = SIZE_TIEBREAK * size / self.budget_bytes

        card = {"error_pct": error, "size_bytes": size, "fits": over == 0,
                "penalty": penalty, "score": error + penalty + tiebreak}
        if len(self._cache) > 300_000:           # keep memory use bounded
            self._cache.clear()
        self._cache[key] = card
        return card

    def score(self, chromosome):
        """Just the number the optimisers minimise. Also counts the evaluation."""
        self.n_evaluations += 1
        return self.score_card(chromosome)["score"]

    def describe(self, chromosome):
        bits, prune = self.decode(chromosome)
        card = self.score_card(chromosome)
        layers = ", ".join(f"L{i + 1}: {bits[i]}-bit/{prune[i]}% pruned"
                           for i in range(N_LAYERS))
        return (f"{layers} | {card['size_bytes']} B "
                f"({'fits' if card['fits'] else 'TOO BIG'}) | "
                f"val error {card['error_pct']:.2f}% | score {card['score']:.4f}")


def load_problem(budget_bytes=FLASH_BUDGET_BYTES):
    """Convenience: load saved model + dataset and build the problem."""
    from model import load_model
    data = np.load("dataset.npz")
    weights, biases = load_model()
    return FlashFitProblem(weights, biases, data["X_val"], data["y_val"],
                           budget_bytes), data
