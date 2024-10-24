from __future__ import annotations

from argparse import ArgumentParser

from polyleven import levenshtein

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammar.synthetic_grammar import create_arbitrary_grammar


def create_target_individual(grammar_seed: int, g: Grammar):
    r = NativeRandomSource(grammar_seed)
    representation = TreeBasedRepresentation(g, decider=MaxDepthDecider(r, g, g.get_min_tree_depth()))
    target_individual = representation.create_genotype(r, depth=10)
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
        target=0,
    )
    r = NativeRandomSource(seed)
    alg = GeneticProgramming(
        problem=problem,
        budget=EvaluationBudget(100),
        representation=TreeBasedRepresentation(g, decider=MaxDepthDecider(r, g, g.get_min_tree_depth() + 10)),
        population_size=10,
        random=r,
        # callbacks=[
        #     ProgressCallback(),
        #     CSVCallback(
        #         filename=filename,
        #         extra_columns={
        #             "non_terminals_count": lambda generation, population, time, gp, ind: non_terminals_count,
        #             "recursive_non_terminals_count": lambda generation, population, time, gp, ind: recursive_non_terminals_count,
        #             "fixed_productions_per_non_terminal": lambda generation, population, time, gp, ind: fixed_productions_per_non_terminal,
        #             "fixed_non_terminals_per_production": lambda generation, population, time, gp, ind: fixed_non_terminals_per_production,
        #         },
        #     ),
        # ],
    )
    ind = alg.search()[0]
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
