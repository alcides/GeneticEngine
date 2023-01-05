from __future__ import annotations

import sys
import random

import numpy as np
import pandas as pd

from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation

from geneticengine.grammars.dynamic_grammar import create_arbitrary_grammar
from geneticengine.grammars.dynamic_grammar import edit_distance

# $ python3 dynamic_grammar_example.py <number_abc_class> <number_terminal_class> <number_nterminal_class> <max_vars_per_class> <number_of_csv_files>
# $ python3 dynamic_grammar_example.py 2 3 3 5 5
# it will create 5 different csv files, with a grammar that contains 2 abstract classes, 3 terminal classes, 3 non-terminal classes, with a maximum of 5 attributes per class


def create_dynamic_grammar(
    grammar_seed,
    n_class_abc,
    n_class_0_children,
    n_class_2_children,
    max_var_per_class,
):

    (list, starting_node) = create_arbitrary_grammar(1, 2, 3)

    return extract_grammar(list, starting_node)


def create_target_individual(grammar_seed: int, g: Grammar):
    r = RandomSource(grammar_seed)
    representation = TreeBasedRepresentation(g, max_depth=g.get_min_tree_depth())
    target_individual = representation.create_individual(r, depth=10)
    individual_phenotype = representation.genotype_to_phenotype(target_individual)
    return individual_phenotype


if __name__ == "__main__":
    # run $ python3 dynamic_grammar_example.py <number_abc_class> <number_terminal_class> <number_nterminal_class> <max_vars_per_class> <number_of_runs>
    args = sys.argv[1:]
    assert len(args) == 5, 'Incorrect input!! It should be e.g: "$ python3 dynamic_grammar_example.py 2 3 3 5 4"'
    seeds_used = [1123]
    for i in range(int(args[-1])):
        # grammar_seed = 100 + i

        # do while
        # generate different random grammar seed each iteration
        while True:
            grammar_seed = random.randint(1, 100 + i)
            if grammar_seed not in seeds_used:
                seeds_used.append(grammar_seed)
                break

        g = create_dynamic_grammar(
            grammar_seed,
            int(args[0]),
            int(args[1]),
            int(args[2]),
            int(args[3]),
        )
        print(g)

        target_individual = create_target_individual(grammar_seed, g)

        def fitness_function(n):
            return edit_distance(str(n), str(target_individual))

        n_generations = 25

        problem = SingleObjectiveProblem(
            fitness_function=fitness_function,
            minimize=True,
            target_fitness=0,
        )

        def evolve(g: Grammar, seed, mode):

            alg = GP(
                representation=TreeBasedRepresentation(g, max_depth=g.get_min_tree_depth() + 10),
                problem=problem,
                population_size=10,
                stopping_criterium=GenerationStoppingCriterium(10),
                random_source=RandomSource(seed),
                callbacks=[
                    CSVCallback(
                        filename=("dynamic_grammar_" + str(grammar_seed)),
                    ),
                ],
            )
            ind = alg.evolve()
            return ind

        ind = evolve(g, 1123, False)
        print(ind)
        print(f"With fitness: {ind.get_fitness(problem)}")

        df = pd.read_csv("dynamic_grammar_" + str(grammar_seed))

        df["abc_classes"] = np.full(n_generations, int(args[0]))
        df["terminal_classes"] = np.full(n_generations, int(args[1]))
        df["non_terminal_classes"] = np.full(n_generations, int(args[2]))

        df.to_csv("dynamic_grammar_" + str(grammar_seed), index=False)
