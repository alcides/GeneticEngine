import numpy as np
from typing import Annotated
from geneticengine.grammar.decorators import weight
from geml.common import GeneticEngineEstimator, PopulationRecorder
from geml.grammars.ruleset_classification import make_grammar
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.hill_climbing import HC
from geneticengine.algorithms.one_plus_one import OnePlusOne
from geneticengine.algorithms.random_search import RandomSearch
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.vars import VarRangeWithProbabilities
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.tree.initializations import ProgressivelyTerminalDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import Individual


class GeneticEngineClassifier(GeneticEngineEstimator):


    def get_grammar(self, feature_names: list[str], data, target) -> Grammar:
        classes = np.unique(target).tolist()
        components, RuleSet = make_grammar(feature_names, classes)
        Var = components[-1]
        weights = self.correlation_weights(feature_names, data, target)
        Var.__init__.__annotations__["name"] = Annotated[str, VarRangeWithProbabilities(feature_names, weights)] # type:ignore
        Var.feature_names = feature_names  # type:ignore
        index_of = {n: i for i, n in enumerate(feature_names)}
        Var.to_numpy = lambda s: f"dataset[:,{index_of[s.name]}]"  # type:ignore
        Var = weight(10)(Var)
        return extract_grammar(components, RuleSet)

    def get_goal(self) -> tuple[bool, float]:
        return True, 1


class GeneticProgrammingClassifier(GeneticEngineClassifier):

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
        return "GPClassifier"


class HillClimbingClassifier(GeneticEngineClassifier):

    def __init__(self, max_time: float | int = 1, seed: int = 0, number_of_mutations: int = 5, weight_features_by_correlation: bool = False):
        super().__init__(max_time, seed, weight_features_by_correlation)
        self.number_of_mutations = number_of_mutations

    _parameter_constraints = {
        "max_time": [float, int],
        "seed": [int],
        "number_of_mutations": [int],
        "weight_features_by_correlation": [bool],
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
        return "HCClassifier"


class RandomSearchClassifier(GeneticEngineClassifier):

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
        return "RSClassifier"


class OnePlusOneClassifier(GeneticEngineClassifier):

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
        return "1+1Classifier"


def model(est, X=None) -> str:
    return est.to_sympy()
