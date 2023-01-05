from __future__ import annotations

from argparse import ArgumentParser

import pandas as pd
from polyleven import levenshtein

from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.synthetic_grammar import create_arbitrary_grammar


def create_target_individual(grammar_seed, g):
    r = RandomSource(grammar_seed)
    representation = treebased_representation
    target_individual = representation.create_individual(r, g, depth=10)
    individual_phenotype = representation.genotype_to_phenotype(g, target_individual)
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
    print(g.get_grammar_specifics())
    target_individual = create_target_individual(seed, g)

    def fitness_function(n):
        "Returns the String difference (levenshtein distance) between the individual and the target"
        return levenshtein(str(n), str(target_individual))

    filename = f"synthetic_grammar_{seed}.csv"
    alg = GP(
        g,
        representation=treebased_representation,
        problem=SingleObjectiveProblem(
            fitness_function=fitness_function,
            minimize=True,
            target_fitness=0,
        ),
        max_depth=g.get_min_tree_depth() + 10,
        population_size=4,
        n_elites=1,
        n_novelties=1,
        number_of_generations=2,
        seed=seed,
        save_to_csv=CSVCallback(
            filename=(filename),
        ),
    )
    (b, bf, bp) = alg.evolve(verbose=2)
    print(b)
    print(f"With fitness: {bf}")

    df = pd.read_csv(filename)
    df["non_terminals_count"] = non_terminals_count
    df["recursive_non_terminals_count"] = recursive_non_terminals_count

    df.to_csv(filename)


if __name__ == "__main__":

    parser = ArgumentParser(
        prog="Synthetic Grammar Example",
    )
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument(
        "-nts",
        "--non_terminals_count",
        dest="non_terminals_count",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-recursives",
        "--recursive_non_terminals_count",
        dest="recursive_non_terminals_count",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-prods",
        "--fixed_productions_per_non_terminal",
        dest="fixed_productions_per_non_terminal",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-nts-per-prod",
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
