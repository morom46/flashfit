# FlashFit

**Fitting a machine-fault classifier into 3 KB of microcontroller flash, using a GA (Genetic Algorithm) to choose how hard to compress each layer.**

A small neural network detects six machine states (healthy, imbalance, misalignment, looseness, two bearing faults) from vibration signals. As trained it needs 29.3 KB. The target is 3 KB. FlashFit searches 8.5 million "recipes" (a bit-width and a pruning level for each of the 4 layers) for the one that fits the budget and loses the least accuracy, and compares the GA against PSO (Particle Swarm Optimisation), SA (Simulated Annealing) and random search.

## Results

All numbers below come from the CSV files in `results/`.

| Model | Size | Fits 3,072 B? | Test error |
|---|---|---|---|
| float32, as trained | 30,040 B | no | 5.83% |
| every layer 4-bit | 4,280 B | no | 6.17% |
| every layer 3-bit | 3,360 B | no | 12.08% |
| every layer 2-bit | 2,440 B | yes | 42.75% |
| best "same in every layer" recipe (3-bit, 60% pruned) | 2,625 B | yes | 8.25% |
| **FlashFit recipe** | **3,052 B** | **yes** | **7.00%** |

- **9.8x smaller** than the float model for **1.17 percentage points** of test accuracy.
- Plain uniform quantisation cannot hit this budget: 3-bit is too big, 2-bit is unusable.
- Against the best recipe a person would try by hand (same setting in every layer), FlashFit is right on 29 test recordings where the hand recipe is wrong, and wrong on 14 where it is right (McNemar's test, p = 0.032).

The winning recipe: layer 1 at 3-bit / 60% pruned, layer 2 at 4-bit / 50%, layer 3 at 4-bit / 60%, layer 4 at 4-bit / unpruned.

### Which optimiser is best?

30 runs each, 12,000 score evaluations per run, different seeds from the ones used for tuning. Score is "lower = better". The true optimum (6.6634) is known because `ground_truth.py` scored every recipe.

| Algorithm | Mean score | Found the true optimum | Welch's t-test vs GA |
|---|---|---|---|
| GA (PyGAD) | 6.7147 | 11 / 30 | - |
| PSO (pyswarms) | 6.7002 | 17 / 30 | p = 0.357, no significant difference |
| SA (scipy dual_annealing) | 6.9216 | 11 / 30 | p = 0.0002, GA better |
| Random search | 7.4924 | 0 / 30 | p < 0.0001, GA better |

Honest reading: GA and PSO are statistically tied on this problem, both clearly beat SA and random search. The GA did not "win".

### Does the compressed model work as a fault detector?

`diagnose.py` turns each prediction into a maintenance decision: any state other than healthy means "needs a check", with advice per fault, and readings below 60% confidence ask for another measurement. On 3,000 new simulated readings (70% healthy, as in a real fleet):

| Model | False alarms (healthy flagged) | Missed faults (fault called healthy) | Right fault type |
|---|---|---|---|
| full, 30,040 B | 153 / 2,127 (7.2%) | 31 / 873 (3.6%) | 821 / 873 |
| compressed, 3,052 B | 153 / 2,127 (7.2%) | 39 / 873 (4.5%) | 808 / 873 |

Same false-alarm rate; the compressed model misses a few more faults, almost all mild imbalance, which looks like the normal once-per-turn vibration of a healthy machine. Of the faults only one model caught, the full model caught 11 and the compressed 3 (p = 0.057, so suggestive but not significant at 0.05). Run with `python diagnose.py --monitor 3000 --seed 1`.

### Diversity

| Variant | Different recipes in final population | Mean score | Hits | vs baseline |
|---|---|---|---|---|
| baseline GA (pop 100) | 99.0 / 100 | 6.7147 | 11 / 30 | - |
| island model (3 x 20) | 52.8 / 60 | 6.7278 | 18 / 30 | p = 0.60 |
| duplicate removal | 100 / 100 | 6.7588 | 8 / 30 | p = 0.072 |

The baseline population was already almost all unique, so removing duplicates had nothing to fix. Neither technique changed the mean score significantly.

### Tuning

One parameter at a time, 10 runs per setting: population 100, mutation probability 0.2, crossover probability 0.9, elitism 1. Switching elitism off was the clearest loser (1/10 hits). Most other differences are inside run-to-run noise - see `results/tuning.csv`.

## Run it

```
python -m pip install -r requirements.txt
python experiments.py demo        # one GA run, about 5 seconds
python diagnose.py                # diagnose one new recording of each state
python -m pytest -q               # 20 unit tests
```

The zip already contains the dataset, the trained model and my results, so `demo` works straight away. To rebuild everything from nothing:

```
python data.py                    # make the synthetic vibration dataset
python model.py                   # train the float model
python ground_truth.py            # optional: score all 8.5M recipes (about 7 min)
python experiments.py all         # tune, compare, diversity, final (about 30-40 min)
```

`python experiments.py compare --runs 5` gives a quick look at any step.

If you re-run `data.py` / `model.py` with different library versions, the trained weights can differ slightly, and every number after that shifts with them. The numbers above belong to the `dataset.npz` and `model_float.npz` in this folder.

## Files

| File | What it does |
|---|---|
| `data.py` | simulates vibration recordings, extracts 40 features, splits train / validation / test |
| `model.py` | trains the MLP (Multi-Layer Perceptron), plain NumPy forward pass |
| `problem.py` | chromosome, quantise, prune, size model, score card |
| `optimisers.py` | `run_ga`, `run_pso`, `run_sa`, `run_random`, `run_islands` |
| `experiments.py` | tuning, comparison, diversity, final report, plots |
| `ground_truth.py` | brute-force search of the whole space |
| `export_header.py` | writes the winning model as a C header |
| `diagnose.py` | uses the compressed model as a fault detector with maintenance advice |
| `tests/test_problem.py` | unit tests |
| `STUDY_GUIDE.md` | how this maps to the coursework, interview questions with answers |

## Method in brief

- **Chromosome:** 8 integers - bits per layer (2, 3, 4, 5, 6, 8) and pruning % per layer (0 to 80 in steps of 10).
- **Score (lower = better):** validation error % + penalty if over budget (5 + 100 x overshoot fraction) + 0.5 x size / budget as a tie-break.
- **Size model:** kept weights x bits, plus a 1-bit-per-weight map if the layer is pruned, plus 32-bit biases.
- **Data split:** 3,000 train (fits the network), 1,800 validation (what the optimisers see), 1,200 test (used once, for the table above).
- **Fair comparison:** same evaluation budget, 30 seeded runs, rivals tuned too, tuning seeds kept separate from comparison seeds.

## Limits

- The vibration data is **synthetic** (generated by formulas in `data.py`), not recorded from a machine.
- Sizes are **calculated** from the storage format above, not measured on a device.
- The C header has **not been compiled or run on a board**, and stores one code per byte; bit-packing is not implemented. It holds weights only; the inference and feature-extraction code exist in Python (`diagnose.py`), not in C.
- The maintenance advice in `diagnose.py` is generic, and the class probabilities are over-confident (the network fits its training data perfectly), so the 60% threshold rarely triggers on the full model.
- Only weights are quantised. Activations stay in float, so RAM use and integer-only inference are not addressed.
- The space is small enough to brute-force. That is deliberate (it gives a ground truth), but it means a metaheuristic is not strictly needed at this size.
- An earlier version used 720 validation recordings; the search over-fitted them (about 6.9% validation error against about 10% on test). Enlarging validation to 1,800 brought the gap under one point.

## Next steps

1. Swap the synthetic signals for a public bearing dataset.
2. Quantisation-aware fine-tuning after the recipe is chosen.
3. Quantise activations and biases; run the header on a real microcontroller and measure flash and latency.
4. Replace the weighted score with a multi-objective search (e.g. NSGA-II) to get the whole size/accuracy trade-off curve.
5. Scale to a network too large to brute-force.
