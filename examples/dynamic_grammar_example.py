from __future__ import annotations

import sys

from geneticengine.algorithms.callbacks.callback import ProgressCallback
from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
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


def run_one_grammar_config(
    seed: int,
    log_file_name: str,
    grammar: Grammar,
    problem: Problem,
    abc_classes: int,
    terminal_classes: int,
    non_terminal_classes: int,
    max_var_per_class: int,
):
    alg = GP(
        representation=TreeBasedRepresentation(grammar, max_depth=10),
        problem=problem,
        stopping_criterium=GenerationStoppingCriterium(25),
        random_source=RandomSource(seed),
        population_size=100,
        callbacks=[
            ProgressCallback(),
            CSVCallback(
                filename=(log_file_name),
                extra_columns={
                    "abc_classes": lambda a, b, c, d, e: abc_classes,
                    "terminal_classes": lambda a, b, c, d, e: terminal_classes,
                    "non_terminal_classes": lambda a, b, c, d, e: non_terminal_classes,
                    "max_var_per_class": lambda a, b, c, d, e: max_var_per_class,
                },
            ),
        ],
    )
    ind = alg.evolve()
    return ind


def generate_and_run_one_grammar_config(
    grammar_seed: int,
    abc_classes: int,
    terminal_classes: int,
    non_terminal_classes: int,
    max_var_per_class: int,
):
    log_file_name = f"dynamic_grammar_{grammar_seed}.csv"
    grammar = create_dynamic_grammar(
        grammar_seed,
        abc_classses,
        terminal_classes,
        non_terminal_classes,
        max_var_per_class,
    )

    target_individual = create_target_individual(grammar_seed, grammar)

    def fitness_function(n):
        return edit_distance(str(n), str(target_individual))

    problem = SingleObjectiveProblem(
        fitness_function=fitness_function,
        minimize=True,
        target_fitness=0,
    )
    return run_one_grammar_config(
        grammar_seed,
        log_file_name,
        grammar,
        problem,
        abc_classses,
        terminal_classes,
        non_terminal_classes,
        max_var_per_class,
    )


if __name__ == "__main__":
    # run $ python3 dynamic_grammar_example.py <number_abc_class> <number_terminal_class> <number_nterminal_class> <max_vars_per_class> <number_of_runs>
    args = sys.argv[1:]
    assert len(args) == 5, 'Incorrect input!! It should be e.g: "$ python3 dynamic_grammar_example.py 2 3 3 5 4"'
    number_of_runs = int(args[-1])
    abc_classses = int(args[0])
    terminal_classes = int(args[1])
    non_terminal_classes = int(args[2])
    max_var_per_class = int(args[3])

    for nrun in range(number_of_runs):
        grammar_seed = nrun
        best = generate_and_run_one_grammar_config(
            grammar_seed,
            abc_classses,
            terminal_classes,
            non_terminal_classes,
            max_var_per_class,
        )
        print(f"{best.fitness} - {best.get_phenotype()}")
