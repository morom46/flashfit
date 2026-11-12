"""
ground_truth.py - (optional, ~5-10 min) score EVERY one of the 8,503,056 recipes.

Why bother? On real problems you never know the true best answer, so you can
only say "algorithm A beat algorithm B". This problem was sized on purpose so
that brute force is just about possible. That gives us the TRUE optimum, and
lets experiments.py report how often each optimiser actually finds it
("hit rate") and how far off it is when it misses ("gap").

Trick that makes it fast: recipes that share the same first layers share the
same intermediate results, so we compute each prefix once (nested loops), and
score all 54 options for the last layer in one NumPy call.

Run:  python ground_truth.py     -> results/ground_truth.json
"""

import itertools
import json
import time

import numpy as np

from problem import (BIT_CHOICES, PRUNE_CHOICES, OVERSHOOT_FIXED,
                     OVERSHOOT_SLOPE, SIZE_TIEBREAK, load_problem)

if __name__ == "__main__":
    problem, _ = load_problem()
    options = list(itertools.product(BIT_CHOICES, PRUNE_CHOICES))   # 54 per layer
    sq = problem._squashed
    X, y, b = problem.X_val, problem.y_val, problem.biases
    budget = problem.budget_bytes

    # Last layer: score all 54 versions together.
    # Side by side they form one wide matrix: (32 inputs, 54 * 6 outputs).
    W4_all = np.hstack([sq[(3, bits, pr)][0] for bits, pr in options])
    b4_all = np.tile(b[3], len(options))
    size4_all = np.array([sq[(3, bits, pr)][1] for bits, pr in options])

    best = []                     # (score, recipe) - we keep the top 20
    n_total = n_fit = 0
    start = time.time()

    for i1, (b1, p1) in enumerate(options):
        a1 = np.maximum(X @ sq[(0, b1, p1)][0] + b[0], 0)
        for b2, p2 in options:
            a2 = np.maximum(a1 @ sq[(1, b2, p2)][0] + b[1], 0)
            for b3, p3 in options:
                a3 = np.maximum(a2 @ sq[(2, b3, p3)][0] + b[2], 0)
                logits = (a3 @ W4_all + b4_all).reshape(len(y), len(options), -1)
                wrong = np.argmax(logits, axis=2) != y[:, None]         # (N, 54)
                error = 100.0 * np.mean(wrong, axis=0)

                bits_total = (sq[(0, b1, p1)][1] + sq[(1, b2, p2)][1]
                              + sq[(2, b3, p3)][1] + size4_all)
                size = np.ceil(bits_total / 8)
                over = np.maximum(0, size - budget) / budget
                penalty = np.where(over == 0, 0.0,
                                   OVERSHOOT_FIXED + OVERSHOOT_SLOPE * over)
                score = error + penalty + SIZE_TIEBREAK * size / budget

                n_total += len(options)
                n_fit += int(np.sum(over == 0))
                k = int(np.argmin(score))
                if len(best) < 20 or score[k] < best[-1][0]:
                    for j in np.argsort(score)[:20]:
                        b4, p4 = options[j]
                        best.append((float(score[j]),
                                     [b1, b2, b3, b4, p1, p2, p3, p4]))
                    best = sorted(best)[:20]
        print(f"{i1 + 1}/54 done, {time.time() - start:.0f}s, "
              f"best so far {best[0][0]:.4f}", flush=True)

    result = {
        "n_recipes": n_total,
        "n_fit_budget": n_fit,
        "optimum_score": best[0][0],
        "optimum_recipe": best[0][1],
        "top20": [{"score": s, "recipe": r} for s, r in best],
    }
    with open("results/ground_truth.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nScored {n_total:,} recipes; {n_fit:,} fit the budget "
          f"({100 * n_fit / n_total:.1f}%).")
    print("True optimum:", problem.describe(best[0][1]))
