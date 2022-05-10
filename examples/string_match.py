from __future__ import annotations

import os

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.grammatical_evolution import ge_representation
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.letter import *


# Extracted from PonyGE
def fit(individual: String):
    guess = str(individual)
    target = "Hello world!"
    fitness: float = max(len(target), len(guess))
    # Loops as long as the shorter of two strings
    for (t_p, g_p) in zip(target, guess):
        if t_p == g_p:
            # Perfect match.
            fitness -= 1
        else:
            # Imperfect match, find ASCII distance to match.
            fitness -= 1 / (1 + (abs(ord(t_p) - ord(g_p))))
    return fitness


def fitness_function(x):
    return fit(x)


def preprocess():
    g = extract_grammar([LetterString, Char, Vowel, Consonant], String)
    return g


def evolve(
    g,
    seed,
    mode,
    representation="treebased_representation",
):
    if representation == "grammatical_evolution":
        representation = ge_representation
    else:
        representation = treebased_representation

    alg = GP(
        g,
        fitness_function,
        representation=representation,
        # As in PonyGE2:
        probability_crossover=0.75,
        probability_mutation=0.01,
        number_of_generations=50,
        max_depth=15,
        # max_init_depth=10,
        population_size=500,
        selection_method=("tournament", 2),
        n_elites=5,
        # ----------------
        minimize=True,
        seed=seed,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
