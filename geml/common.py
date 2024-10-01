from typing import Any

import numpy as np

from geneticengine.evaluation.recorder import SearchRecorder
from geneticengine.problems import Problem
from geneticengine.solutions.individual import Individual


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


def forward_dataset(e: Any, dataset) -> float:
    with np.errstate(all="ignore"):
        r = eval(e.to_numpy(), {"dataset": dataset, "np": np})
    return wrap_in_shape(r, right_shape=(len(dataset), 1))


def PredictorWrapper(BaseEstimator):
    def __init__(self, ind: Individual):
        self.ind = ind

    def predict(self, X):
        feature_names, data = self.prepare_inputs(X)
        return forward_dataset(self.ind.get_phenotype(), data)
