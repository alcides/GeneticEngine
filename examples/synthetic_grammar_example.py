from __future__ import annotations

from argparse import ArgumentParser

from geneticengine.algorithms.callbacks.callback import ProgressCallback
from geneticengine.algorithms.gp.operators.stop import (
    AnyOfStoppingCriterium,
    SingleFitnessTargetStoppingCriterium,
    GenerationStoppingCriterium,
)
from polyleven import levenshtein

from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import Grammar, extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammars.synthetic_grammar import create_arbitrary_grammar


def create_target_individual(grammar_seed: int, g: Grammar):
    r = RandomSource(grammar_seed)
    representation = TreeBasedRepresentation(g, max_depth=g.get_min_tree_depth())
    target_individual = representation.create_individual(r, depth=10)
    individual_phenotype = representation.genotype_to_phenotype(target_individual)
    return individual_phenotype


def single_run(
    seed: int,
    non_terminals_count: int,
    recursive_non_terminals_count: int,
    fixed_productions_per_non_terminal: int,
    fixed_non_terminals_per_production: int,
):
    (nodes, root) = create_arbitrary_grammar(
        seed=seed,
        non_terminals_count=non_terminals_count,
        recursive_non_terminals_count=recursive_non_terminals_count,
        productions_per_non_terminal=lambda rd: fixed_productions_per_non_terminal,
        non_terminals_per_production=lambda rd: fixed_non_terminals_per_production,
    )
    g = extract_grammar(nodes, root)
    print(g)
    target_individual = create_target_individual(seed, g)

    def fitness_function(n):
        "Returns the String difference (levenshtein distance) between the individual and the target"
        return levenshtein(str(n), str(target_individual))

    problem = SingleObjectiveProblem(
        fitness_function=fitness_function,
        minimize=True,
    )

    stopping_criterium = AnyOfStoppingCriterium(
        GenerationStoppingCriterium(10),
        SingleFitnessTargetStoppingCriterium(0),
    )

    filename = f"synthetic_grammar_{seed}.csv"
    alg = GP(
        representation=TreeBasedRepresentation(g, max_depth=g.get_min_tree_depth() + 10),
        problem=problem,
        population_size=10,
        stopping_criterium=stopping_criterium,
        random_source=RandomSource(seed),
        callbacks=[
            ProgressCallback(),
            CSVCallback(
                filename=filename,
                extra_columns={
                    "non_terminals_count": lambda generation, population, time, gp, ind: non_terminals_count,
                    "recursive_non_terminals_count": lambda generation, population, time, gp, ind: recursive_non_terminals_count,
                    "fixed_productions_per_non_terminal": lambda generation, population, time, gp, ind: fixed_productions_per_non_terminal,
                    "fixed_non_terminals_per_production": lambda generation, population, time, gp, ind: fixed_non_terminals_per_production,
                },
            ),
        ],
    )
    ind = alg.evolve()
    print(ind)
    print(f"With fitness: {ind.get_fitness(problem)}")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Synthetic Grammar Example",
    )
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument(
        "--non_terminals_count",
        dest="non_terminals_count",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--recursive_non_terminals_count",
        dest="recursive_non_terminals_count",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--fixed_productions_per_non_terminal",
        dest="fixed_productions_per_non_terminal",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--fixed_non_terminals_per_production",
        dest="fixed_non_terminals_per_production",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    single_run(
        args.seed,
        args.non_terminals_count,
        args.recursive_non_terminals_count,
        args.fixed_productions_per_non_terminal,
        args.fixed_non_terminals_per_production,
    )
