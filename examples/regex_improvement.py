from geneticengine.grammars.regex import *

from examples.regex_fitness.RegexEval import *

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation


# Extracted from PonyGE
def fit(individual: RE):
    regexeval: RegexEval = RegexEval()
    return regexeval(individual)


fitness_function = lambda x: fit(x)

if __name__ == "__main__":
    g = extract_grammar([
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
    ], RE)
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        max_depth=100,
        population_size=1000,
        n_elites=100,
        number_of_generations=100,
        probability_crossover=0.5,
        selection_method=("tournament", 2),
        minimize=True,
    )
    print("Started running...")
    (b, bf, bp) = alg.evolve(verbose=0)
    print(f"Best individual: {b}")
    print(f"With fitness: {bf}")
