from abc import ABC, abstractmethod
from typing import Annotated, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score
from geml.common import PopulationRecorder, PredictorWrapper, forward_dataset
from geml.grammars.symbolic_regression import Var, components, Expression
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.hill_climbing import HC
from geneticengine.algorithms.one_plus_one import OnePlusOne
from geneticengine.algorithms.random_search import RandomSearch
from geneticengine.evaluation.budget import TimeBudget
from geneticengine.evaluation.tracker import SingleObjectiveProgressTracker
from geneticengine.grammar.decorators import weight
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.vars import VarRange
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource, RandomSource
from geneticengine.representations.tree.initializations import ProgressivelyTerminalDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import Individual


class SRBenchRegressor(ABC):
    max_time: int

    def __init__(self, max_time: int):
        self.max_time = max_time

    @abstractmethod
    def get_population(self) -> list[BaseEstimator]: ...

    @abstractmethod
    def get_best_individual(self) -> BaseEstimator: ...


class GeneticEngineRegressor(
    BaseEstimator,
    TransformerMixin,
    SRBenchRegressor,
):

    random: RandomSource
    best_individual: Individual
    best_individuals: list[Individual]

    def __init__(self, max_time: int, seed: int = 0, remove_time_overheads=True):
        if remove_time_overheads:
            max_time = max_time - (60 * 60)
        SRBenchRegressor.__init__(self, max_time)

        self.random = NativeRandomSource(seed)

    def prepare_inputs(self, X) -> tuple[list[str], Any]:
        if isinstance(X, pd.DataFrame):
            return list(X.columns.values), X.values
        else:
            return [f"x{i}" for i in range(X.shape[1])], X

    def prepare_outputs(self, y) -> Any:
        if isinstance(y, pd.Series):
            return y.values
        else:
            return y

    def predict(self, X):
        feature_names, data = self.prepare_inputs(X)
        return forward_dataset(self.best_individual.get_phenotype(), data)

    def get_population(self) -> list[str]:
        return [PredictorWrapper(x) for x in self.best_individuals]

    def get_best_individual(self) -> str:
        return PredictorWrapper(self.best_individual)

    def fit(self, X, y):
        feature_names, data = self.prepare_inputs(X)
        target = self.prepare_outputs(y)
        assert data.shape[0] == target.shape[0]

        Var.__init__.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
        Var.feature_names = feature_names
        index_of = {n: i for i, n in enumerate(feature_names)}
        Var.to_numpy = lambda s: f"dataset[:,{index_of[s.name]}]"
        complete_components = components + [weight(10)(Var)]
        self.grammar: Grammar = extract_grammar(complete_components, Expression)

        def fitness_function(x: Expression) -> float:
            try:
                y_pred = forward_dataset(x, data)
                with np.errstate(all="ignore"):
                    return r2_score(target, y_pred)
            except ValueError:
                return -10000000

        self.problem = SingleObjectiveProblem(fitness_function)

        self.population_recorder = PopulationRecorder()

        self.best_individual = self.search()
        self.best_individuals = self.population_recorder.best_individuals

    def to_sympy(self):
        return self.best_individual.get_phenotype().to_sympy()

    @abstractmethod
    def search(self) -> Individual: ...


class GeneticProgrammingRegressor(GeneticEngineRegressor):

    def search(self) -> Individual:
        decider = ProgressivelyTerminalDecider(self.random, self.grammar)
        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(self.grammar, decider),
            problem=self.problem,
            random=self.random,
            budget=TimeBudget(self.max_time),
            tracker=SingleObjectiveProgressTracker(problem=self.problem, recorders=[self.population_recorder]),
        )
        return gp.search()

    def __str__(self):
        return "GPRegressor"


class HillClimbingRegressor(GeneticEngineRegressor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_mutations = kwargs.get("number_of_mutations", 5)

    def search(self) -> Individual:
        decider = ProgressivelyTerminalDecider(self.random, self.grammar)
        hc = HC(
            representation=TreeBasedRepresentation(self.grammar, decider),
            problem=self.problem,
            random=self.random,
            budget=TimeBudget(self.max_time),
            tracker=SingleObjectiveProgressTracker(problem=self.problem, recorders=[self.population_recorder]),
            number_of_mutations=self.number_of_mutations,
        )
        return hc.search()

    def __str__(self):
        return "HCRegressor"


class RandomSearchRegressor(GeneticEngineRegressor):

    def search(self) -> Individual:
        decider = ProgressivelyTerminalDecider(self.random, self.grammar)
        hc = RandomSearch(
            representation=TreeBasedRepresentation(self.grammar, decider),
            problem=self.problem,
            random=self.random,
            budget=TimeBudget(self.max_time),
            tracker=SingleObjectiveProgressTracker(problem=self.problem, recorders=[self.population_recorder]),
        )
        return hc.search()

    def __str__(self):
        return "RSRegressor"


class OnePlusOneRegressor(GeneticEngineRegressor):

    def search(self) -> Individual:
        decider = ProgressivelyTerminalDecider(self.random, self.grammar)
        hc = OnePlusOne(
            representation=TreeBasedRepresentation(self.grammar, decider),
            problem=self.problem,
            random=self.random,
            budget=TimeBudget(self.max_time),
            tracker=SingleObjectiveProgressTracker(problem=self.problem, recorders=[self.population_recorder]),
        )
        return hc.search()

    def __str__(self):
        return "1+1Regressor"


def model(est, X=None) -> str:
    return est.to_sympy()
