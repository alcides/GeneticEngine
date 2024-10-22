from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from geml.common import GeneticEngineEstimator, PopulationRecorder
from geml.grammars.symbolic_regression import make_var, components, Expression
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.hill_climbing import HC
from geneticengine.algorithms.one_plus_one import OnePlusOne
from geneticengine.algorithms.random_search import RandomSearch
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.tree.initializations import ProgressivelyTerminalDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import Individual


class SRBenchRegressor(ABC):
    max_time: float | int

    def __init__(self, max_time: float | int):
        self.max_time = max_time

    @abstractmethod
    def get_population(self) -> list[BaseEstimator]: ...

    @abstractmethod
    def get_best_solution(self) -> BaseEstimator: ...


class GeneticEngineRegressor(
    GeneticEngineEstimator,
):

    def get_grammar(self, feature_names: list[str], data, target) -> Grammar:
        Var = make_var(feature_names, relative_weight=10)
        Var.feature_names = feature_names
        index_of = {n: i for i, n in enumerate(feature_names)}
        Var.to_numpy = lambda s: f"dataset[:,{index_of[s.name]}]"
        complete_components = components + [Var]
        return extract_grammar(complete_components, Expression)

    def get_goal(self) -> tuple[bool, float]:
        return False, 0


class GeneticProgrammingRegressor(GeneticEngineRegressor):

    def search(
        self,
        grammar: Grammar,
        problem: Problem,
        random: RandomSource,
        budget: SearchBudget,
        population_recorder: PopulationRecorder,
    ) -> list[Individual] | None:
        decider = ProgressivelyTerminalDecider(random, grammar)
        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(grammar, decider),
            problem=problem,
            random=random,
            budget=budget,
            tracker=ProgressTracker(problem=problem, recorders=[population_recorder]),
        )
        return gp.search()

    def __str__(self):
        return "GPRegressor"


class HillClimbingRegressor(GeneticEngineRegressor):

    def __init__(self, max_time: float | int = 1, seed: int = 0, number_of_mutations: int = 5):
        super().__init__(max_time, seed)
        self.number_of_mutations = number_of_mutations

    _parameter_constraints = {
        "max_time": [float, int],
        "seed": [int],
        "number_of_mutations": [int],
    }

    def search(
        self,
        grammar: Grammar,
        problem: Problem,
        random: RandomSource,
        budget: SearchBudget,
        population_recorder: PopulationRecorder,
    ) -> list[Individual] | None:
        decider = ProgressivelyTerminalDecider(random, grammar)
        hc = HC(
            representation=TreeBasedRepresentation(grammar, decider),
            problem=problem,
            random=random,
            budget=budget,
            tracker=ProgressTracker(problem=problem, recorders=[population_recorder]),
        )
        return hc.search()

    def __str__(self):
        return "HCRegressor"


class RandomSearchRegressor(GeneticEngineRegressor):

    def search(
        self,
        grammar: Grammar,
        problem: Problem,
        random: RandomSource,
        budget: SearchBudget,
        population_recorder: PopulationRecorder,
    ) -> list[Individual] | None:
        decider = ProgressivelyTerminalDecider(random, grammar)
        rs = RandomSearch(
            representation=TreeBasedRepresentation(grammar, decider),
            problem=problem,
            random=random,
            budget=budget,
            tracker=ProgressTracker(problem=problem, recorders=[population_recorder]),
        )
        return rs.search()

    def __str__(self):
        return "RSRegressor"


class OnePlusOneRegressor(GeneticEngineRegressor):

    def search(
        self,
        grammar: Grammar,
        problem: Problem,
        random: RandomSource,
        budget: SearchBudget,
        population_recorder: PopulationRecorder,
    ) -> list[Individual] | None:
        decider = ProgressivelyTerminalDecider(random, grammar)
        hc = OnePlusOne(
            representation=TreeBasedRepresentation(grammar, decider),
            problem=problem,
            random=random,
            budget=budget,
            tracker=ProgressTracker(problem=problem, recorders=[population_recorder]),
        )
        return hc.search()

    def __str__(self):
        return "1+1Regressor"


def model(est, X=None) -> str:
    if est is None:
        return "0"
    else:
        return est.to_sympy()
