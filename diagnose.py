"""
diagnose.py - Step 7: use the compressed model as a working fault detector.

Everything before this file only MEASURED the compressed model (how often is
it right?). This file USES it: take one vibration recording, decide what state
the machine is in, and say whether it needs a check.

    python diagnose.py                       one recording of every state
    python diagnose.py --state inner_race    one recording of one state
    python diagnose.py --monitor 600         simulate a day of readings and
                                             count alarms, false alarms and
                                             missed faults

Each recording is NEW: made with seeds the dataset never used, so the model
has not seen it before. It is still SIMULATED by data.py, not read from a
real sensor.
"""

import argparse
import json

import numpy as np

from data import CLASS_NAMES, extract_features, make_signal
from problem import N_LAYERS, load_problem

HEALTHY = 0
CONFIDENCE_THRESHOLD = 0.60     # below this, ask for another reading
NEW_SEED_BASE = 900_000         # far away from the seed the dataset used

ADVICE = {
    "healthy": "No action needed.",
    "imbalance": "Check rotor balance (build-up on the impeller, missing balance weight).",
    "misalignment": "Check shaft alignment between motor and pump.",
    "looseness": "Check mounting bolts, base and bearing housing fit.",
    "outer_race": "Bearing fault suspected (outer race): schedule a bearing inspection.",
    "inner_race": "Bearing fault suspected (inner race): schedule a bearing inspection.",
}


# --------------------------------------------------------------------------
# The detector
# --------------------------------------------------------------------------
def softmax(logits):
    """Turn the network's raw outputs into probabilities that add up to 1."""
    shifted = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def decide(probabilities):
    """Turn 6 probabilities into a maintenance decision (a dict).

    needs_check is True whenever the most likely state is not healthy.
    If the model is not sure enough, it says so instead of guessing.
    """
    label = int(np.argmax(probabilities))
    confidence = float(probabilities[label])
    state = CLASS_NAMES[label]
    if confidence < CONFIDENCE_THRESHOLD:
        message = ("Uncertain reading: take another measurement "
                   f"(best guess {state}, {confidence:.0%}).")
    else:
        message = ADVICE[state]
    return {"label": label, "state": state, "confidence": confidence,
            "needs_check": label != HEALTHY,
            "uncertain": confidence < CONFIDENCE_THRESHOLD,
            "message": message}


class Detector:
    """Runs recordings through either the full model or the compressed one."""

    def __init__(self, compressed=True):
        problem, data = load_problem()
        if compressed:
            recipe = json.load(open("results/best_recipe.json"))["recipe"]
            self.weights = problem.squashed_weights(recipe)
            self.size_bytes = problem.size_bytes(recipe)
        else:
            self.weights = problem.weights
            self.size_bytes = sum(W.size + b.size for W, b in
                                  zip(problem.weights, problem.biases)) * 4
        self.biases = problem.biases
        # The features must be scaled exactly as the training data was.
        self.mean, self.std = data["feat_mean"], data["feat_std"]

    def probabilities(self, signals):
        """signals: a list of raw recordings -> (n, 6) probabilities."""
        X = np.array([extract_features(s) for s in signals])
        a = ((X - self.mean) / self.std).astype(np.float32)
        for i in range(N_LAYERS):
            a = a @ self.weights[i] + self.biases[i]
            if i < N_LAYERS - 1:
                a = np.maximum(a, 0)                # ReLU on hidden layers
        return softmax(a)

    def diagnose(self, signal):
        return decide(self.probabilities([signal])[0])


# --------------------------------------------------------------------------
# Demo modes
# --------------------------------------------------------------------------
def show(true_state, full, small):
    for name, result in (("full model  ", full), ("compressed  ", small)):
        flag = "CHECK" if result["needs_check"] else "ok   "
        print(f"  {name} [{flag}] {result['state']:13s} "
              f"{result['confidence']:5.0%}  {result['message']}")
    agree = "agree" if full["state"] == small["state"] else "DISAGREE"
    correct = "correct" if small["state"] == true_state else "WRONG"
    print(f"  -> models {agree}; compressed model is {correct}\n")


def one_of_each(full_model, small_model, seed):
    print(f"Full model {full_model.size_bytes:,} B | "
          f"compressed model {small_model.size_bytes:,} B\n")
    for label, state in enumerate(CLASS_NAMES):
        rng = np.random.default_rng(NEW_SEED_BASE + seed * 10 + label)
        signal = make_signal(label, rng)
        print(f"Machine really is: {state}")
        show(state, full_model.diagnose(signal), small_model.diagnose(signal))


def monitor(full_model, small_model, n, seed):
    """Simulate n readings from a fleet where most machines are healthy.

    In a factory most readings are healthy, so we use 70% healthy and split
    the rest over the five faults. Two mistakes matter:
      false alarm  - healthy machine flagged: wasted engineer visit
      missed fault - broken machine called healthy: the expensive one
    """
    rng = np.random.default_rng(NEW_SEED_BASE + 500_000 + seed)
    weights = [0.70] + [0.06] * 5
    labels = rng.choice(len(CLASS_NAMES), size=n, p=weights)
    signals = [make_signal(int(lab), rng) for lab in labels]
    faulty = labels != HEALTHY

    print(f"{n} simulated readings: {int((~faulty).sum())} healthy, "
          f"{int(faulty.sum())} faulty\n")
    print(f"{'model':12s} {'alarms':>7s} {'false alarms':>13s} "
          f"{'missed faults':>14s} {'right fault type':>17s} {'uncertain':>10s}")
    for name, model in (("full", full_model), ("compressed", small_model)):
        results = [decide(p) for p in model.probabilities(signals)]
        flagged = np.array([r["needs_check"] for r in results])
        predicted = np.array([r["label"] for r in results])
        uncertain = sum(r["uncertain"] for r in results)
        false_alarms = int(np.sum(flagged & ~faulty))
        missed = int(np.sum(~flagged & faulty))
        right_type = int(np.sum((predicted == labels) & faulty))
        print(f"{name:12s} {int(flagged.sum()):7d} "
              f"{false_alarms:6d} ({false_alarms / max(1, (~faulty).sum()):5.1%}) "
              f"{missed:6d} ({missed / max(1, faulty.sum()):5.1%}) "
              f"{right_type:8d} / {int(faulty.sum()):<6d} {uncertain:7d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashFit fault detector demo")
    parser.add_argument("--state", choices=CLASS_NAMES,
                        help="diagnose one recording of this state")
    parser.add_argument("--monitor", type=int, metavar="N",
                        help="simulate N readings and count alarms")
    parser.add_argument("--seed", type=int, default=0,
                        help="change this to get different recordings")
    args = parser.parse_args()

    full_model, small_model = Detector(compressed=False), Detector(compressed=True)
    if args.monitor:
        monitor(full_model, small_model, args.monitor, args.seed)
    elif args.state:
        label = CLASS_NAMES.index(args.state)
        rng = np.random.default_rng(NEW_SEED_BASE + args.seed * 10 + label)
        signal = make_signal(label, rng)
        print(f"Machine really is: {args.state}")
        show(args.state, full_model.diagnose(signal), small_model.diagnose(signal))
    else:
        one_of_each(full_model, small_model, args.seed)
