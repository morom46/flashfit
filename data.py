"""
data.py - Step 1: make the (synthetic) dataset.

What this file does
-------------------
A vibration sensor (accelerometer) is bolted to a motor/pump. Different faults
leave different "fingerprints" in the vibration signal. We simulate those
signals, turn each one into 40 numbers (features), and save train / validation
/ test splits.

Six machine states (classes):
    0 healthy        - just a small once-per-turn vibration + noise
    1 imbalance      - strong vibration at 1x the shaft speed
    2 misalignment   - strong vibration at 2x the shaft speed
    3 looseness      - many harmonics (1x, 2x, 3x ...) plus a half-speed tone
    4 outer race     - bearing fault: regular "clicks" that ring a resonance
    5 inner race     - bearing fault: clicks whose strength rises and falls
                       once per shaft turn

The data is SYNTHETIC (made by formulas, not recorded from a real machine).
That is stated openly in the README; swapping in a real dataset is a listed
next step.

Run:  python data.py
"""

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

FS = 6400            # sampling rate in Hz (samples per second)
N_SAMPLES = 2048     # samples per recording (0.32 seconds)
CLASS_NAMES = ["healthy", "imbalance", "misalignment",
               "looseness", "outer_race", "inner_race"]

# Bearing geometry constants: how many clicks per shaft turn each fault makes.
BPFO = 3.57          # Ball Pass Frequency Outer race (multiples of shaft speed)
BPFI = 5.43          # Ball Pass Frequency Inner race


def make_signal(label, rng):
    """Return one vibration recording (1-D array) for the given class label."""
    t = np.arange(N_SAMPLES) / FS
    shaft_hz = rng.uniform(25, 35)            # machine speed changes run to run
    severity = rng.uniform(0.4, 1.0)         # how bad the fault is
    phase = rng.uniform(0, 2 * np.pi, size=8)

    # Every machine has a little 1x vibration, even when healthy.
    x = 0.30 * np.sin(2 * np.pi * shaft_hz * t + phase[0])
    x += 0.08 * np.sin(2 * np.pi * 2 * shaft_hz * t + phase[1])

    if label == 1:                            # imbalance -> big 1x
        x += 1.0 * severity * np.sin(2 * np.pi * shaft_hz * t + phase[2])

    elif label == 2:                          # misalignment -> big 2x, some 3x
        x += 0.9 * severity * np.sin(2 * np.pi * 2 * shaft_hz * t + phase[2])
        x += 0.3 * severity * np.sin(2 * np.pi * 3 * shaft_hz * t + phase[3])

    elif label == 3:                          # looseness -> many harmonics
        for k in range(1, 9):
            x += (0.5 * severity / k) * np.sin(2 * np.pi * k * shaft_hz * t + phase[k - 1])
        x += 0.25 * severity * np.sin(2 * np.pi * 0.5 * shaft_hz * t)

    elif label in (4, 5):                     # bearing faults -> clicks
        click_hz = (BPFO if label == 4 else BPFI) * shaft_hz
        resonance_hz = rng.uniform(2200, 2800)   # the "bell" the clicks ring
        click_times = np.arange(rng.uniform(0, 1 / click_hz), t[-1], 1 / click_hz)
        for ct in click_times:
            strength = 2.0 * severity
            if label == 5:                    # inner race passes through the
                # load zone once per turn, so click strength goes up and down
                strength *= 0.5 * (1 + np.sin(2 * np.pi * shaft_hz * ct))
            after = t >= ct
            x[after] += (strength * np.exp(-900 * (t[after] - ct))
                         * np.sin(2 * np.pi * resonance_hz * (t[after] - ct)))

    # Sensor noise (the level also varies, to keep the task realistic).
    x += rng.normal(0, rng.uniform(0.20, 0.40), size=N_SAMPLES)
    return x


def band_energies(spectrum, n_bands, lo_bin, hi_bin):
    """Split a spectrum into n_bands log-spaced bands; return log energy of each."""
    edges = np.unique(np.geomspace(lo_bin, hi_bin, n_bands + 1).astype(int))
    # np.unique can merge very narrow bands; pad edges so we always get n_bands
    while len(edges) < n_bands + 1:
        edges = np.append(edges, edges[-1] + 1)
    energies = [np.sum(spectrum[edges[i]:edges[i + 1]] ** 2) for i in range(n_bands)]
    return np.log10(np.array(energies) + 1e-9)


def extract_features(x):
    """Turn one recording into 40 numbers a small neural network can use.

    24 = energy in 24 frequency bands of the raw spectrum
    12 = energy in 12 bands of the ENVELOPE spectrum (the classic way to find
         bearing clicks: track the loudness outline, then look at its rhythm)
     4 = RMS (overall level), kurtosis (spikiness), crest factor, peak-to-peak
    """
    window = np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x * window))
    raw_bands = band_energies(spectrum, 24, 2, len(spectrum) - 1)

    envelope = np.abs(hilbert(x))
    env_spectrum = np.abs(np.fft.rfft((envelope - envelope.mean()) * window))
    env_bands = band_energies(env_spectrum, 12, 2, 160)   # up to ~500 Hz

    rms = np.sqrt(np.mean(x ** 2))
    stats = np.array([rms, kurtosis(x), np.max(np.abs(x)) / rms, np.ptp(x)])
    return np.concatenate([raw_bands, env_bands, stats])


def build_dataset(n_per_class=1000, seed=0):
    """Make features X and labels y, then split 50 / 30 / 20.

    train      -> used to train the neural network
    validation -> used by the optimisers to score each compression recipe
    test       -> touched ONCE at the very end, to report honest accuracy
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in range(len(CLASS_NAMES)):
        for _ in range(n_per_class):
            X.append(extract_features(make_signal(label, rng)))
            y.append(label)
    X, y = np.array(X), np.array(y)

    order = rng.permutation(len(y))           # shuffle before splitting
    X, y = X[order], y[order]
    n_train, n_val = int(0.5 * len(y)), int(0.3 * len(y))

    # Standardise using TRAIN statistics only (no peeking at val/test).
    mean, std = X[:n_train].mean(axis=0), X[:n_train].std(axis=0) + 1e-9
    X = (X - mean) / std

    return {
        "X_train": X[:n_train], "y_train": y[:n_train],
        "X_val": X[n_train:n_train + n_val], "y_val": y[n_train:n_train + n_val],
        "X_test": X[n_train + n_val:], "y_test": y[n_train + n_val:],
        "feat_mean": mean, "feat_std": std,
    }


if __name__ == "__main__":
    data = build_dataset()
    np.savez("dataset.npz", **data)
    for part in ("train", "val", "test"):
        print(f"{part:5s}: {data['X_' + part].shape[0]} recordings x "
              f"{data['X_' + part].shape[1]} features")
    print("Saved dataset.npz")
