"""
Unit tests: small checks that each building block does what it claims.

Run from the project folder:   python -m pytest -q

They use a tiny made-up network, so they do not need dataset.npz and finish
in a couple of seconds.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimisers import decode_continuous, random_recipe, run_ga, run_random
from problem import (BIT_CHOICES, GENE_SPACE, PRUNE_CHOICES, FlashFitProblem,
                     layer_size_bits, prune_weights, quantise_weights)


@pytest.fixture
def tiny_problem():
    """A 5 -> 6 -> 5 -> 4 -> 3 network with random weights and 60 samples."""
    rng = np.random.default_rng(0)
    shapes = [(5, 6), (6, 5), (5, 4), (4, 3)]
    weights = [rng.normal(size=s) for s in shapes]
    biases = [rng.normal(size=s[1]) for s in shapes]
    X = rng.normal(size=(60, 5))
    y = rng.integers(0, 3, size=60)
    return FlashFitProblem(weights, biases, X, y, budget_bytes=120)


# ---- quantise ---------------------------------------------------------------
def test_two_bit_quantise_gives_only_three_levels():
    W = np.random.default_rng(1).normal(size=(10, 10))
    codes, scale = quantise_weights(W, 2)
    assert set(np.unique(codes)) <= {-1, 0, 1}
    assert scale > 0


def test_more_bits_means_smaller_rounding_error():
    W = np.random.default_rng(2).normal(size=(20, 20))
    errors = []
    for bits in BIT_CHOICES:
        codes, scale = quantise_weights(W, bits)
        errors.append(np.mean(np.abs(codes * scale - W)))
    assert errors == sorted(errors, reverse=True)      # 2-bit worst, 8-bit best


def test_codes_fit_in_int8():
    W = np.random.default_rng(3).normal(size=(20, 20)) * 50
    codes, _ = quantise_weights(W, 8)
    assert codes.min() >= -127 and codes.max() <= 127


# ---- prune ------------------------------------------------------------------
def test_prune_zero_changes_nothing():
    W = np.random.default_rng(4).normal(size=(8, 8))
    assert np.array_equal(prune_weights(W, 0), W)


def test_prune_removes_the_smallest_weights():
    W = np.arange(1, 11, dtype=float).reshape(2, 5)     # 1, 2, ... 10
    pruned = prune_weights(W, 50)
    assert np.count_nonzero(pruned) == 5
    assert set(pruned[pruned != 0]) == {6, 7, 8, 9, 10}


# ---- size model -------------------------------------------------------------
def test_layer_size_by_hand():
    # 100 weights, 4 biases, 4-bit, no pruning: 100*4 + 0 + 4*32 = 528 bits
    assert layer_size_bits(100, 4, 4, 0, 100) == 528
    # same layer, 50% pruned: 50*4 + 100 (map) + 128 = 428 bits
    assert layer_size_bits(100, 4, 4, 50, 50) == 428


def test_light_pruning_at_low_bits_makes_the_layer_bigger():
    unpruned = layer_size_bits(100, 0, 2, 0, 100)       # 200 bits
    pruned_10 = layer_size_bits(100, 0, 2, 10, 90)      # 180 + 100 = 280 bits
    assert pruned_10 > unpruned


# ---- score card -------------------------------------------------------------
def test_recipe_that_fits_has_no_penalty(tiny_problem):
    card = tiny_problem.score_card([2, 2, 2, 2, 80, 80, 80, 80])
    assert card["fits"] and card["penalty"] == 0
    expected = card["error_pct"] + 0.5 * card["size_bytes"] / 120
    assert card["score"] == pytest.approx(expected)


def test_recipe_over_budget_is_penalised(tiny_problem):
    card = tiny_problem.score_card([8, 8, 8, 8, 0, 0, 0, 0])
    assert not card["fits"]
    assert card["penalty"] >= 5.0
    # Any too-big recipe must score worse than its own error alone + 5.
    assert card["score"] > card["error_pct"] + 5.0


def test_bigger_overshoot_means_bigger_penalty(tiny_problem):
    small_over = tiny_problem.score_card([6, 6, 6, 6, 0, 0, 0, 0])["penalty"]
    big_over = tiny_problem.score_card([8, 8, 8, 8, 0, 0, 0, 0])["penalty"]
    assert big_over > small_over


def test_score_counts_evaluations(tiny_problem):
    recipe = [4, 4, 4, 4, 0, 0, 0, 0]
    tiny_problem.score(recipe)
    tiny_problem.score(recipe)                           # cached, still counted
    assert tiny_problem.n_evaluations == 2


# ---- optimisers -------------------------------------------------------------
def test_decode_continuous_hits_both_ends():
    assert decode_continuous([0.0] * 8) == [2, 2, 2, 2, 0, 0, 0, 0]
    assert decode_continuous([1.0] * 8) == [8, 8, 8, 8, 80, 80, 80, 80]


def test_random_recipe_is_always_legal():
    rng = np.random.default_rng(5)
    for _ in range(200):
        recipe = random_recipe(rng)
        assert all(gene in choices for gene, choices in zip(recipe, GENE_SPACE))


def test_ga_returns_legal_recipe_and_is_repeatable(tiny_problem):
    first = run_ga(tiny_problem, pop=10, gens=5, seed=7)
    second = run_ga(tiny_problem, pop=10, gens=5, seed=7)
    assert first["best_recipe"] == second["best_recipe"]            # same seed
    assert all(g in c for g, c in zip(first["best_recipe"], GENE_SPACE))
    assert first["best_recipe"][0] in BIT_CHOICES
    assert first["best_recipe"][4] in PRUNE_CHOICES


def test_history_never_gets_worse(tiny_problem):
    history = run_random(tiny_problem, seed=0)["history"][:500]
    assert all(later <= earlier for earlier, later in zip(history, history[1:]))


# ---- diagnose (the fault detector) -------------------------------------------
from diagnose import ADVICE, CONFIDENCE_THRESHOLD, decide, softmax
from data import CLASS_NAMES


def test_softmax_gives_probabilities():
    p = softmax(np.array([2.0, 1.0, 0.1, -1.0, 0.0, 3.0]))
    assert p.sum() == pytest.approx(1.0)
    assert np.argmax(p) == 5 and np.all(p > 0)


def test_healthy_reading_needs_no_check():
    result = decide(np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01]))
    assert result["state"] == "healthy"
    assert not result["needs_check"]
    assert result["message"] == ADVICE["healthy"]


def test_fault_reading_needs_a_check():
    result = decide(np.array([0.02, 0.02, 0.02, 0.02, 0.90, 0.02]))
    assert result["state"] == "outer_race" and result["needs_check"]
    assert "bearing" in result["message"].lower()


def test_unsure_reading_asks_for_another_measurement():
    unsure = CONFIDENCE_THRESHOLD - 0.1
    rest = (1 - unsure) / 5
    result = decide(np.array([rest, unsure, rest, rest, rest, rest]))
    assert result["uncertain"]
    assert "another measurement" in result["message"]


def test_every_state_has_advice():
    assert set(ADVICE) == set(CLASS_NAMES)
