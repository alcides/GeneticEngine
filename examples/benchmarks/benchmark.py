from abc import ABC

from geml.common import PopulationRecorder
from geneticengine.algorithms.enumerative import EnumerativeSearch
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget, TimeBudget
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import ProgressivelyTerminalDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


class Benchmark(ABC):
    def get_problem(self) -> Problem: ...

    def get_grammar(self) -> Grammar: ...


def example_run(b: Benchmark):
    problem = b.get_problem()
    grammar = b.get_grammar()
    random = NativeRandomSource(1)
    alg = GeneticProgramming(
        problem=problem,
        budget=TimeBudget(5),
        representation=TreeBasedRepresentation(
            grammar,
            decider=ProgressivelyTerminalDecider(random=random, grammar=grammar),
        ),
        random=random,
        tracker=ProgressTracker(problem, recorders=[PopulationRecorder()]),
    )
    best = alg.search()[0]
    print(
        f"Fitness of {best.get_fitness(problem)} by genotype: {best} with phenotype: {best.get_phenotype()}",
    )

def example_run_enumerative(b: Benchmark):
    problem = b.get_problem()
    grammar = b.get_grammar()
    alg = EnumerativeSearch(
        problem=problem,
        budget=EvaluationBudget(1000000),
        grammar= grammar,
        tracker=ProgressTracker(problem, recorders=[PopulationRecorder()]),
    )
    best = alg.search()[0]
    print(
        f"Fitness of {best.get_fitness(problem)} by phenotype: {best.get_phenotype()}",
    )
