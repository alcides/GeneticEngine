from abc import ABC
from typing import Any

from geml.common import PopulationRecorder
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import TimeBudget
from geneticengine.evaluation.recorder import SearchRecorder
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import ProgressivelyTerminalDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import Individual


class DebugRecorder(SearchRecorder):
    def __init__(self, slots=100):
        self.best_individuals = []
        self.slots = slots

    def register(self, tracker: Any, individual: Individual, problem: Problem, is_best: bool):
        print(individual.get_fitness(problem).maximizing_aggregate, individual.get_phenotype())
        # if is_best:
        #    print(individual.get_fitness(problem).maximizing_aggregate, individual.get_phenotype())


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
        tracker=ProgressTracker(problem, recorders=[PopulationRecorder(), DebugRecorder()]),
    )
    best = alg.search()[0]
    print(
        f"Fitness of {best.get_fitness(problem)} with phenotype: {best.get_phenotype()}",
    )
