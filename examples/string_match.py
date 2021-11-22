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

if __name__ == "__main__":
    g = extract_grammar([LetterString, Char, Vowel, Consonant], String)
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        max_depth=17,
        probability_crossover=0.75,
        selection_method=("tournament", 2),
        minimize=True,
    )
    print("Started running...")
    (b, bf, bp) = alg.evolve(verbose=0)
    print(f"Best individual: {b}")
    print(f"With fitness: {bf}")
