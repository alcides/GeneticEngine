from geneticengine.grammars.letter import *
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation


# Extracted from PonyGE
def fit(individual: String):
    guess = str(individual)
    target = 'Hello world!'
    fitness = max(len(target), len(guess))
    # Loops as long as the shorter of two strings
    for (t_p, g_p) in zip(target, guess):
        if t_p == g_p:
            # Perfect match.
            fitness -= 1
        else:
            # Imperfect match, find ASCII distance to match.
            fitness -= 1 / (1 + (abs(ord(t_p) - ord(g_p))))
    return fitness


fitness_function = lambda x: fit(x)


def preprocess():
    g = extract_grammar([LetterString, Char, Vowel, Consonant], String)
    return g


def evolve(g, seed, mode):
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        max_depth=17,
        probability_crossover=0.75,
        selection_method=("tournament", 2),
        minimize=True,
        seed=seed,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve(verbose=0)
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    evolve(g, 1, True)
