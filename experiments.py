"""
experiments.py - Step 5: run the experiments and write the results.

    python experiments.py demo        one GA run + fitness plot          (~5 s)
    python experiments.py tune        one-parameter-at-a-time tuning     (long)
    python experiments.py compare     GA vs PSO vs SA vs random search   (long)
    python experiments.py diversity   baseline vs islands vs no-twins    (long)
    python experiments.py final       test-set report, plot, C header    (~5 s)
    python experiments.py all         everything above, in order

Add  --runs 5  for a quick look. The reported results use the default 30
(tuning uses 10 repeats per setting, as in the course recipe).

Everything is written to the results/ folder as CSV / PNG / JSON.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # save plots to files, no window
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from optimisers import BUDGET, run_ga, run_islands, run_pso, run_random, run_sa
from problem import BIT_CHOICES, N_LAYERS, PRUNE_CHOICES, load_problem

RESULTS = Path("results")
ALPHA = 0.05                              # significance threshold
TUNE_RUNS = 10                            # repeats per setting while tuning
                                          # (the course recipe uses 10)
TUNE_SEED_OFFSET = 1000                   # tuning uses different seeds from the
                                          # final comparison (no "marking your
                                          # own homework")
DEFAULT_CONFIG = {"ga": {"pop": 60, "mut_prob": 0.1, "cx_prob": 0.7, "elitism": 1},
                  "pso": {"w": 0.7}, "sa": {"initial_temp": 5230.0}}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def load_optimum():
    """True best score from ground_truth.py, or None if it has not been run."""
    path = RESULTS / "ground_truth.json"
    return json.loads(path.read_text())["optimum_score"] if path.exists() else None


def load_config():
    path = RESULTS / "tuned_config.json"
    return json.loads(path.read_text()) if path.exists() else DEFAULT_CONFIG


def repeat(run_fn, runs, seed_offset=0, **settings):
    """Run an optimiser `runs` times with seeds 0, 1, 2 ... and collect results."""
    return [run_fn(seed=seed_offset + i, **settings) for i in range(runs)]


def summarise(results, optimum=None):
    """Mean / spread / best of the final scores, plus hit rate if optimum known."""
    scores = np.array([r["best_score"] for r in results])
    row = {"mean": scores.mean(), "std": scores.std(ddof=1),
           "median": np.median(scores), "best": scores.min(),
           "worst": scores.max(),
           "mean_evals": np.mean([r["evaluations"] for r in results])}
    if optimum is not None:
        row["hits"] = int(np.sum(scores <= optimum + 1e-9))
        row["mean_gap"] = scores.mean() - optimum
    return row


def write_csv(name, rows):
    with open(RESULTS / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> results/{name}")


def compare_two(name_a, a, name_b, b):
    """Welch's t-test (the course method) + Mann-Whitney U (a second opinion
    that does not assume bell-shaped data). Lower score = better."""
    t, p_welch = stats.ttest_ind(a, b, equal_var=False)
    _, p_mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
    if p_welch < ALPHA:
        verdict = f"{name_a if np.mean(a) < np.mean(b) else name_b} significantly better"
    else:
        verdict = "no significant difference"
    print(f"  {name_a} vs {name_b}: t = {t:+.3f}, Welch p = {p_welch:.4f}, "
          f"Mann-Whitney p = {p_mwu:.4f} -> {verdict}")
    return {"a": name_a, "b": name_b, "mean_a": np.mean(a), "mean_b": np.mean(b),
            "t": t, "p_welch": p_welch, "p_mannwhitney": p_mwu, "verdict": verdict}


def fmt(row):
    text = f"mean {row['mean']:.4f}  std {row['std']:.4f}  best {row['best']:.4f}"
    if "hits" in row:
        text += f"  hits {row['hits']}"
    return text


# --------------------------------------------------------------------------
# demo - one GA run
# --------------------------------------------------------------------------
def demo(problem, runs):
    result = run_ga(problem, seed=0, **load_config()["ga"])
    print("Best recipe:", problem.describe(result["best_recipe"]))
    print("Evaluations used:", result["evaluations"])

    plt.figure(figsize=(7, 4))
    plt.plot(result["history"])
    plt.xlabel("score evaluations")
    plt.ylabel("best score so far (lower = better)")
    plt.title("FlashFit GA - one run")
    plt.ylim(top=min(result["history"]) + 6)
    plt.tight_layout()
    plt.savefig(RESULTS / "fitness_plot.png", dpi=150)
    plt.close()
    print("  -> results/fitness_plot.png")


# --------------------------------------------------------------------------
# tune - one parameter at a time (param_div.txt recipe)
# --------------------------------------------------------------------------
def tune(problem, runs):
    optimum = load_optimum()
    rows = []

    def sweep(algo, run_fn, settings, name, values):
        """Try each value for one parameter, keep the one with the best mean."""
        best_value, best_mean = None, float("inf")
        for value in values:
            trial = dict(settings, **{name: value})
            row = summarise(repeat(run_fn, min(runs, TUNE_RUNS), TUNE_SEED_OFFSET,
                                   problem=problem, **trial), optimum)
            rows.append({"algorithm": algo, "parameter": name, "value": value, **row})
            print(f"  {algo} {name} = {value}: {fmt(row)}")
            if row["mean"] < best_mean:
                best_value, best_mean = value, row["mean"]
        settings[name] = best_value
        print(f"  => keep {name} = {best_value}\n")

    # GA: population first (generations follow from the fixed budget:
    # gens = BUDGET // pop, so a bigger population gets fewer generations).
    ga = dict(DEFAULT_CONFIG["ga"])
    sweep("GA", run_ga, ga, "pop", [20, 40, 60, 100])
    sweep("GA", run_ga, ga, "mut_prob", [0.05, 0.1, 0.2])
    sweep("GA", run_ga, ga, "cx_prob", [0.5, 0.7, 0.9])
    sweep("GA", run_ga, ga, "elitism", [0, 1, 2])

    # The rivals get a (smaller) tuning pass too, so the comparison is not
    # "tuned GA vs untuned everything else".
    pso = dict(DEFAULT_CONFIG["pso"])
    sweep("PSO", run_pso, pso, "w", [0.5, 0.7, 0.9])
    sa = dict(DEFAULT_CONFIG["sa"])
    sweep("SA", run_sa, sa, "initial_temp", [100.0, 1000.0, 5230.0])

    config = {"ga": ga, "pso": pso, "sa": sa}
    (RESULTS / "tuned_config.json").write_text(json.dumps(config, indent=2))
    write_csv("tuning.csv", rows)
    print("Tuned settings:", config)


# --------------------------------------------------------------------------
# compare - GA vs PSO vs SA vs random search
# --------------------------------------------------------------------------
def compare(problem, runs):
    config, optimum = load_config(), load_optimum()
    results = {
        "GA": repeat(run_ga, runs, problem=problem, **config["ga"]),
        "PSO": repeat(run_pso, runs, problem=problem, **config["pso"]),
        "SA": repeat(run_sa, runs, problem=problem, **config["sa"]),
        "Random": repeat(run_random, runs, problem=problem),
    }
    scores = {name: [r["best_score"] for r in rs] for name, rs in results.items()}

    print("Summary (lower = better):")
    summary = []
    for name, rs in results.items():
        row = summarise(rs, optimum)
        summary.append({"algorithm": name, **row})
        print(f"  {name:6s} {fmt(row)}")
    write_csv("comparison_summary.csv", summary)
    write_csv("comparison_runs.csv",
              [{"run": i, **{n: s[i] for n, s in scores.items()}} for i in range(runs)])

    print(f"Hypothesis tests (null: same mean score; alpha = {ALPHA}):")
    tests = [compare_two("GA", scores["GA"], other, scores[other])
             for other in ("PSO", "SA", "Random")]
    write_csv("comparison_stats.csv", tests)

    # Best recipe found by anyone (judged on validation score only).
    best_algo, best_run = min(((n, r) for n, rs in results.items() for r in rs),
                              key=lambda pair: pair[1]["best_score"])
    (RESULTS / "best_recipe.json").write_text(json.dumps(
        {"found_by": best_algo, "recipe": best_run["best_recipe"],
         "score": best_run["best_score"]}, indent=2))
    print(f"Best recipe (found by {best_algo}):",
          problem.describe(best_run["best_recipe"]))

    # Plot 1: spread of final scores.
    plt.figure(figsize=(7, 4))
    plt.boxplot(list(scores.values()), tick_labels=list(scores.keys()))
    if optimum is not None:
        plt.axhline(optimum, linestyle="--", color="grey", label="true optimum")
        plt.legend()
    plt.ylabel("final score (lower = better)")
    plt.title(f"{runs} runs each, {BUDGET} evaluations per run")
    plt.tight_layout()
    plt.savefig(RESULTS / "comparison_boxplot.png", dpi=150)
    plt.close()

    # Plot 2: how fast each one gets there (mean best-so-far).
    plt.figure(figsize=(7, 4))
    for name, rs in results.items():
        curves = np.array([np.pad(r["history"], (0, BUDGET - len(r["history"][:BUDGET])),
                                  mode="edge")[:BUDGET] for r in rs])
        plt.plot(curves.mean(axis=0), label=name)
    if optimum is not None:
        plt.axhline(optimum, linestyle="--", color="grey")
        plt.ylim(optimum - 0.1, optimum + 3)
    plt.xscale("log")
    plt.xlabel("score evaluations (log scale)")
    plt.ylabel("mean best score so far")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "convergence.png", dpi=150)
    plt.close()
    print("  -> results/comparison_boxplot.png, results/convergence.png")


# --------------------------------------------------------------------------
# diversity - does keeping the population varied help?
# --------------------------------------------------------------------------
def diversity(problem, runs):
    ga, optimum = load_config()["ga"], load_optimum()
    island_settings = {k: ga[k] for k in ("mut_prob", "cx_prob", "elitism")}
    variants = {
        "baseline GA": repeat(run_ga, runs, problem=problem, **ga),
        "island model": repeat(run_islands, runs, problem=problem, **island_settings),
        "no duplicates": repeat(run_ga, runs, problem=problem,
                                remove_duplicates=True, **ga),
    }

    rows = []
    for name, rs in variants.items():
        # Diversity measure: how many DIFFERENT recipes are in the final population.
        unique = [len({tuple(r) for r in run["final_population"]}) for run in rs]
        size = len(rs[0]["final_population"])
        row = summarise(rs, optimum)
        rows.append({"variant": name, "mean_unique": np.mean(unique),
                     "population": size, **row})
        print(f"  {name:14s} unique recipes at the end: {np.mean(unique):.1f}/{size}"
              f"  | {fmt(row)}")
    write_csv("diversity.csv", rows)

    base = [r["best_score"] for r in variants["baseline GA"]]
    tests = [compare_two("baseline GA", base, name, [r["best_score"] for r in rs])
             for name, rs in variants.items() if name != "baseline GA"]
    write_csv("diversity_stats.csv", tests)


# --------------------------------------------------------------------------
# final - honest report on the untouched test set
# --------------------------------------------------------------------------
def final(problem, data, runs):
    from export_header import export_header
    from model import accuracy

    X_test, y_test = data["X_test"], data["y_test"]
    path = RESULTS / "best_recipe.json"
    if path.exists():
        best = json.loads(path.read_text())
    else:                                  # compare not run yet: do one GA run
        run = run_ga(problem, seed=0, **load_config()["ga"])
        best = {"found_by": "GA", "recipe": run["best_recipe"]}

    float_bytes = sum(W.size + b.size for W, b in zip(problem.weights, problem.biases)) * 4
    float_err_val = 100 * (1 - accuracy(problem.weights, problem.biases,
                                        problem.X_val, problem.y_val))
    float_err_test = 100 * (1 - accuracy(problem.weights, problem.biases,
                                         X_test.astype(np.float32), y_test))
    rows = [{"model": "float32 (uncompressed)", "size_bytes": float_bytes,
             "fits_budget": False, "val_error_pct": float_err_val,
             "test_error_pct": float_err_test}]

    candidates = [(f"uniform {b}-bit, no pruning", [b] * N_LAYERS + [0] * N_LAYERS)
                  for b in reversed(BIT_CHOICES)]

    # Fair "by hand" baseline: the same bits AND the same pruning in every
    # layer (54 options). Keep the best one that fits, judged on validation.
    by_hand = [[b] * N_LAYERS + [p] * N_LAYERS
               for b in BIT_CHOICES for p in PRUNE_CHOICES]
    best_by_hand = min(by_hand, key=lambda r: problem.score_card(r)["score"])
    bits, prune = best_by_hand[0], best_by_hand[N_LAYERS]
    candidates.append((f"best same-everywhere ({bits}-bit/{prune}%)", best_by_hand))
    candidates.append((f"FlashFit recipe ({best['found_by']})", best["recipe"]))
    for name, recipe in candidates:
        card = problem.score_card(recipe)
        rows.append({"model": name, "size_bytes": card["size_bytes"],
                     "fits_budget": card["fits"], "val_error_pct": card["error_pct"],
                     "test_error_pct": problem.error_pct(recipe, X_test, y_test)})

    print(f"{'model':32s} {'bytes':>7s} {'fits':>5s} {'val err':>8s} {'test err':>9s}")
    for r in rows:
        print(f"{r['model']:32s} {r['size_bytes']:7d} {str(r['fits_budget']):>5s} "
              f"{r['val_error_pct']:7.2f}% {r['test_error_pct']:8.2f}%")
    write_csv("final_report.csv", rows)

    # Is the gain over the by-hand recipe real? McNemar's test: look only at
    # test recordings where exactly ONE of the two models is right.
    ours_right = problem.predict(best["recipe"], X_test) == y_test
    hand_right = problem.predict(best_by_hand, X_test) == y_test
    only_ours = int(np.sum(ours_right & ~hand_right))
    only_hand = int(np.sum(~ours_right & hand_right))
    p = stats.binomtest(only_ours, only_ours + only_hand, 0.5).pvalue
    print(f"FlashFit vs best same-everywhere on the test set: right where the "
          f"other is wrong {only_ours} vs {only_hand}, McNemar p = {p:.4f}")
    write_csv("final_mcnemar.csv", [{"only_flashfit_right": only_ours,
                                     "only_by_hand_right": only_hand, "p": p}])

    # Plot: size against test accuracy, with the budget line.
    plt.figure(figsize=(8, 4.5))
    for i, r in enumerate(rows):
        chosen = r["model"].startswith("FlashFit")
        label = (r["model"].split(" (")[0].replace("uniform ", "")
                 .replace(", no pruning", "").replace("best same-everywhere", "by hand"))
        plt.scatter(r["size_bytes"] / 1024, 100 - r["test_error_pct"],
                    marker="*" if chosen else "o", s=220 if chosen else 50,
                    color="tab:red" if chosen else "tab:blue", zorder=3)
        # alternate labels above / below so neighbours do not overlap
        plt.annotate(label, (r["size_bytes"] / 1024, 100 - r["test_error_pct"]),
                     textcoords="offset points", ha="center", fontsize=8,
                     xytext=(0, 10) if (i % 2 == 0 or chosen) else (0, -16))
    plt.axvline(problem.budget_bytes / 1024, linestyle="--", color="grey")
    plt.text(problem.budget_bytes / 1024, 60, " flash budget", color="grey")
    plt.xscale("log")
    plt.xticks([2.5, 3, 4, 6, 8, 16, 30], ["2.5", "3", "4", "6", "8", "16", "30"])
    plt.minorticks_off()
    plt.ylim(50, 100)
    plt.xlabel("model size (KB, log scale)")
    plt.ylabel("test accuracy (%)")
    plt.title("Size vs accuracy on the held-out test set")
    plt.tight_layout()
    plt.savefig(RESULTS / "size_vs_accuracy.png", dpi=150)
    plt.close()
    print("  -> results/size_vs_accuracy.png")

    export_header(problem, best["recipe"], RESULTS / "flashfit_model.h")
    print("  -> results/flashfit_model.h")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashFit experiments")
    parser.add_argument("step", choices=["demo", "tune", "compare", "diversity",
                                         "final", "all"])
    parser.add_argument("--runs", type=int, default=30,
                        help="repeats per setting (default 30)")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    problem, data = load_problem()
    steps = {"demo": demo, "tune": tune, "compare": compare, "diversity": diversity,
             "final": lambda p, r: final(p, data, r)}
    for step in (["demo", "tune", "compare", "diversity", "final"]
                 if args.step == "all" else [args.step]):
        print(f"\n===== {step} =====")
        steps[step](problem, args.runs)
