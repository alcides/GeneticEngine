from __future__ import annotations

from examples.archived.regex_fitness.RegexEval import RegexEval
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.regex import *


# Extracted from PonyGE
def fit(individual: RE):
    regexeval: RegexEval = RegexEval()
    return regexeval.evaluate(individual)


def fitness_function(x):
    return fit(x)


def preprocess():
    return extract_grammar(
        [
            ElementaryREParens,
            ElementaryREWD,
            ElementaryRERE,
            ModifierSingle,
            ModifierOr,
            LookaroundSingle,
            LookaroundComposition,
            Char,
            Set,
            RangeAnChar1,
            RangeAnChar2,
            RangeLimits,
            RecurDigitSingle,
            RecurDigitMultiple,
            MatchTimesSingleRecur,
            MatchTimesDoubleRecur,
        ],
        RE,
    )


def evolve(g, seed, mode):
    alg = GP(
        g,
        fitness_function,
        representation=treebased_representation,
        max_depth=100,
        population_size=1000,
        n_elites=100,
        number_of_generations=100,
        probability_crossover=0.5,
        selection_method=("tournament", 2),
        minimize=True,
        seed=seed,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve(verbose=2)


if __name__ == "__main__":
    g = preprocess()
    b, bf = evolve(g, 0, False)
    print(f"Best individual: {b}")
    print(f"With fitness: {bf}")
