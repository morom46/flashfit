"""
optimisers.py - Step 4: the search algorithms.

Every function here answers the same question ("which recipe has the lowest
score?") with the same budget of score evaluations, and returns the same kind
of result dict, so they can be compared fairly.

    run_ga       GA  (Genetic Algorithm)            via PyGAD
    run_pso      PSO (Particle Swarm Optimisation)  via pyswarms
    run_sa       SA  (Simulated Annealing)          via scipy dual_annealing
    run_random   random search - the "is any of this clever?" baseline
    run_islands  island-model GA (several small populations that swap members)
"""

import numpy as np
import pygad
import pyswarms
from scipy.optimize import dual_annealing

from problem import GENE_SPACE, N_GENES

BUDGET = 12000        # score evaluations each algorithm is allowed per run


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
class Tracker:
    """Wraps problem.score so every algorithm is measured the same way.

    Remembers the best recipe seen and the best-so-far score after every
    single evaluation (used for the convergence plot).
    """

    def __init__(self, problem):
        self.problem = problem
        self.best_score = float("inf")
        self.best_recipe = None
        self.history = []                 # best-so-far after each evaluation

    def score(self, recipe):
        s = self.problem.score(recipe)
        if s < self.best_score:
            self.best_score, self.best_recipe = s, [int(g) for g in recipe]
        self.history.append(self.best_score)
        return s

    def result(self, **extra):
        out = {"best_score": self.best_score, "best_recipe": self.best_recipe,
               "history": self.history, "evaluations": len(self.history)}
        out.update(extra)
        return out


def random_recipe(rng):
    """One random chromosome: pick a legal value for each of the 8 genes."""
    return [int(rng.choice(choices)) for choices in GENE_SPACE]


def decode_continuous(x):
    """PSO and SA work with decimals between 0 and 1, not with our gene values.

    This maps each decimal to a slot in that gene's list of choices.
    Example: 6 bit choices -> 0.00-0.166 is slot 0 (2-bit), 0.166-0.333 is
    slot 1 (3-bit) ... (the same idea as rounding to 5-minute blocks).
    """
    recipe = []
    for value, choices in zip(x, GENE_SPACE):
        slot = int(np.clip(value, 0, 0.999999) * len(choices))
        recipe.append(choices[slot])
    return recipe


# --------------------------------------------------------------------------
# GA (Genetic Algorithm) with PyGAD
# --------------------------------------------------------------------------
def run_ga(problem, pop=60, gens=None, mut_prob=0.1, cx_prob=0.7, elitism=1,
           seed=0, remove_duplicates=False, initial_population=None):
    """One GA run.

    pop      population size (recipes alive at once)
    gens     generations; default = BUDGET // pop so the budget stays equal
    mut_prob chance that EACH gene of a child is reset to a random legal value
    cx_prob  chance that two parents are mixed (single-point crossover)
    elitism  how many of the best recipes are copied unchanged to the next
             generation
    remove_duplicates  replace identical twins with fresh random recipes
    """
    gens = BUDGET // pop if gens is None else gens
    tracker = Tracker(problem)
    rng = np.random.default_rng(seed)

    def fitness_func(ga, solution, solution_idx):
        return -tracker.score(solution)       # PyGAD maximises, we minimise

    def on_generation(ga):
        if not remove_duplicates:
            return
        seen = set()
        for i, recipe in enumerate(ga.population):
            key = tuple(int(g) for g in recipe)
            if key in seen:                   # a twin -> swap in a newcomer
                newcomer = random_recipe(rng)
                ga.population[i] = newcomer
                # PyGAD keeps last generation's fitness for parent selection,
                # so the newcomer's fitness must be written in too, otherwise
                # it would be judged with the twin's old fitness.
                ga.last_generation_fitness[i] = -tracker.score(newcomer)
                key = tuple(newcomer)
            seen.add(key)

    ga = pygad.GA(
        num_generations=gens,
        sol_per_pop=pop,
        num_genes=N_GENES,
        gene_space=GENE_SPACE,                # legal values for every gene
        gene_type=int,
        initial_population=initial_population,
        fitness_func=fitness_func,
        num_parents_mating=max(2, pop // 2),
        parent_selection_type="tournament", K_tournament=3,
        crossover_type="single_point", crossover_probability=cx_prob,
        mutation_type="random", mutation_probability=mut_prob,
        keep_elitism=elitism,
        keep_parents=-1 if elitism > 0 else 0,   # elitism 0 = keep nobody
        on_generation=on_generation,
        random_seed=seed,
        suppress_warnings=True,
    )
    ga.run()
    final_population = [[int(g) for g in recipe] for recipe in ga.population]
    return tracker.result(final_population=final_population)


# --------------------------------------------------------------------------
# PSO (Particle Swarm Optimisation) with pyswarms
# --------------------------------------------------------------------------
def run_pso(problem, particles=60, w=0.7, c1=1.5, c2=1.5, seed=0):
    """One PSO run: particles fly through [0,1]^8 and are decoded to recipes.

    w  inertia (keep going the way you were going)
    c1 pull towards the particle's own best spot
    c2 pull towards the swarm's best spot
    """
    tracker = Tracker(problem)
    np.random.seed(seed)                      # pyswarms uses NumPy's global RNG

    def cost(positions):                      # positions: (particles, 8)
        return np.array([tracker.score(decode_continuous(x)) for x in positions])

    swarm = pyswarms.single.GlobalBestPSO(
        n_particles=particles, dimensions=N_GENES,
        options={"w": w, "c1": c1, "c2": c2},
        bounds=(np.zeros(N_GENES), np.ones(N_GENES)))
    swarm.optimize(cost, iters=BUDGET // particles, verbose=False)
    return tracker.result()


# --------------------------------------------------------------------------
# SA (Simulated Annealing) with scipy
# --------------------------------------------------------------------------
def run_sa(problem, initial_temp=5230.0, seed=0):
    """One SA run: a single recipe takes random steps; bad steps are sometimes
    accepted while the "temperature" is high, less and less as it cools."""
    tracker = Tracker(problem)

    def cost(x):
        if len(tracker.history) >= BUDGET:    # hard stop exactly at the budget
            return tracker.best_score
        return tracker.score(decode_continuous(x))

    dual_annealing(cost, bounds=[(0, 1)] * N_GENES, maxfun=BUDGET,
                   initial_temp=initial_temp, no_local_search=True, seed=seed)
    return tracker.result()


# --------------------------------------------------------------------------
# Random search (baseline)
# --------------------------------------------------------------------------
def run_random(problem, seed=0):
    """Try BUDGET random recipes and keep the best. No learning at all."""
    tracker = Tracker(problem)
    rng = np.random.default_rng(seed)
    for _ in range(BUDGET):
        tracker.score(random_recipe(rng))
    return tracker.result()


# --------------------------------------------------------------------------
# Island-model GA (diversity technique)
# --------------------------------------------------------------------------
def run_islands(problem, n_islands=3, island_pop=20, rounds=8, mut_prob=0.1,
                cx_prob=0.7, elitism=1, seed=0):
    """Several small GAs evolve separately; after each round the best recipe of
    each island moves to the next island in a ring (0 -> 1 -> 2 -> 0)."""
    gens_per_round = BUDGET // (n_islands * island_pop * rounds)
    rng = np.random.default_rng(seed)
    islands = [[random_recipe(rng) for _ in range(island_pop)]
               for _ in range(n_islands)]
    best_score, best_recipe, history = float("inf"), None, []

    for round_no in range(rounds):
        for i in range(n_islands):
            run = run_ga(problem, pop=island_pop, gens=gens_per_round,
                         mut_prob=mut_prob, cx_prob=cx_prob, elitism=elitism,
                         seed=seed * 1000 + round_no * 10 + i,
                         initial_population=islands[i])
            islands[i] = run["final_population"]
            for s in run["history"]:          # stitch the histories together
                history.append(min(s, best_score))
            if run["best_score"] < best_score:
                best_score, best_recipe = run["best_score"], run["best_recipe"]

        # Ring migration: best of island i replaces worst of island i + 1.
        scores = [[problem.score_card(r)["score"] for r in isl] for isl in islands]
        migrants = [islands[i][int(np.argmin(scores[i]))] for i in range(n_islands)]
        for i in range(n_islands):
            target = (i + 1) % n_islands
            islands[target][int(np.argmax(scores[target]))] = list(migrants[i])

    final_population = [r for isl in islands for r in isl]
    return {"best_score": best_score, "best_recipe": best_recipe,
            "history": history, "evaluations": len(history),
            "final_population": final_population}
