from abc import abstractmethod
from typing import Any

import numpy as np

import pandas as pd
from sklearn.base import BaseEstimator, check_is_fitted, _fit_context
from sklearn.metrics import r2_score

from geneticengine.evaluation.budget import SearchBudget, TimeBudget
from geneticengine.evaluation.recorder import SearchRecorder
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.solutions.individual import Individual

from geml.grammars.symbolic_regression import Expression
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource


class PopulationRecorder(SearchRecorder):
    def __init__(self, slots=100):
        self.best_individuals = []
        self.slots = slots

    def register(self, tracker: Any, individual: Individual, problem: Problem, is_best: bool):
        if is_best:
            self.best_individuals.insert(0, individual)
            if len(self.best_individuals) > self.slots:
                self.best_individuals.pop()


def wrap_in_shape(r, right_shape: tuple[int, int]):
    if not isinstance(r, np.ndarray):
        return np.full(shape=right_shape, fill_value=r)
    elif len(r.shape) == 0:
        return np.full(shape=right_shape, fill_value=0)
    elif r.shape[0] != right_shape[0]:
        return np.full(shape=right_shape, fill_value=r)
    else:
        return r.reshape(right_shape)


def forward_dataset(e: str, dataset) -> float:
    assert isinstance(e, str)
    with np.errstate(all="ignore"):
        r = eval(e, {"dataset": dataset, "np": np})
    return wrap_in_shape(r, right_shape=(len(dataset), 1))


def PredictorWrapper(BaseEstimator):
    def __init__(self, ind: tuple[str, str]):
        self.ind = ind

    def predict(self, X):
        _, data = self.prepare_inputs(X)
        return forward_dataset(self.ind[0], data)

    def to_sympy(self):
        return self.ind[1]


class GeneticEngineEstimator(BaseEstimator):
    max_time: float | int

    def __init__(self, max_time: float | int = 1, seed: int = 0):
        self.max_time = max_time
        self.seed = 0

    _parameter_constraints = {
        "max_time": [float, int],
        "seed": [int],
    }

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

    def get_population(self) -> list[str]:
        return [PredictorWrapper(x) for x in self._best_individuals]

    def get_best_individual(self) -> str:
        return PredictorWrapper(self._best_individual)

    def get_budget(self) -> SearchBudget:
        return TimeBudget(self.max_time)

    def to_sympy(self):
        if hasattr(self, "_best_individual"):
            return self._best_individual[1]
        else:
            return "0"

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)
        feature_names, data = self.prepare_inputs(X)
        return forward_dataset(self._best_individual[0], data)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):

        X, y = self._validate_data(X, y, accept_sparse=False)

        random = NativeRandomSource(self.seed)

        feature_names, data = self.prepare_inputs(X)
        target = self.prepare_outputs(y)
        assert data.shape[0] == target.shape[0]

        grammar = self.get_grammar(feature_names, data, target)

        def fitness_function(x: Expression) -> float:
            try:
                y_pred = forward_dataset(x.to_numpy(), data)
                with np.errstate(all="ignore"):
                    return r2_score(target, y_pred)
            except ValueError:
                return -10000000

        problem = SingleObjectiveProblem(fitness_function)

        population_recorder = PopulationRecorder()

        best_individual = self.search(grammar, problem, random, self.get_budget(), population_recorder)
        assert best_individual is not None, "Best individual is none..."

        def make_pair(ind: Individual) -> tuple[str, str]:
            return (
                ind.get_phenotype().to_numpy(),
                ind.get_phenotype().to_sympy(),
            )

        self._best_individual = make_pair(best_individual)

        self._best_individuals = [make_pair(ind) for ind in population_recorder.best_individuals]

        self.is_fitted_ = True
        return self

    @abstractmethod
    def get_grammar(self, feature_names: list[str], data, target) -> Grammar: ...

    @abstractmethod
    def search(
        self,
        grammar: Grammar,
        problem: Problem,
        random: RandomSource,
        budget: SearchBudget,
        population_recorder: PopulationRecorder,
    ) -> Individual: ...