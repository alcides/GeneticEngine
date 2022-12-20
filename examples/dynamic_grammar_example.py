from __future__ import annotations

import random
import sys

from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammars.dynamic_grammar import create_grammar_nodes
from geneticengine.grammars.dynamic_grammar import edit_distance

# $ python3 dynamic_grammar_example.py <number_abc_class> <number_terminal_class> <number_nterminal_class> <max_vars_per_class> <number_of_csv_files>
# $ python3 dynamic_grammar_example.py 2 3 3 5 5
# it will create 5 different csv files, with a grammar that contains 2 abstract classes, 3 terminal classes, 3 non-terminal classes, with a maximum of 5 attributes per class


def create_dynamic_grammar(grammar_seed, n_class_abc, n_class_0_children, n_class_2_children, max_var_per_class):
    (list, starting_node) = create_grammar_nodes(
        grammar_seed,
        n_class_abc,
        n_class_0_children,
        n_class_2_children,
        max_var_per_class,
    )

    return extract_grammar(list, starting_node)


def create_target_individual(grammar_seed, g):
    r = RandomSource(grammar_seed)
    representation = TreeBasedRepresentation(g, max_depth=10)
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

        log_file_name = f"dynamic_grammar_{grammar_seed}.csv"

        g = create_dynamic_grammar(grammar_seed, int(args[0]), int(args[1]), int(args[2]), int(args[3]))
        print(g)

        target_individual = create_target_individual(grammar_seed, g)

        def fitness_function(n):
            return edit_distance(str(n), str(target_individual))

        n_generations = 25

        def evolve(g, seed, mode):
            alg = GP(
                representation=TreeBasedRepresentation(g, max_depth=10),
                problem=SingleObjectiveProblem(
                    fitness_function=fitness_function,
                    minimize=True,
                    target_fitness=0,
                ),
                stopping_criterium=GenerationStoppingCriterium(n_generations),
                random_source=RandomSource(seed),
                population_size=100,
                callbacks=[
                    CSVCallback(
                        filename=(log_file_name),
                        extra_columns={
                            "abc_classes": lambda a, b, c, d, e: int(args[0]),
                            "terminal_classes": lambda a, b, c, d, e: int(args[1]),
                            "non_terminal_classes": lambda a, b, c, d, e: int(args[2]),
                        },
                    ),
                ],
            )
            (b, bf, bp) = alg.evolve()
            return b, bf

        bf, b = evolve(g, 1123, False)
        print(b)
        print(f"With fitness: {bf}")
