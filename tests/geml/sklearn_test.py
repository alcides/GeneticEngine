from __future__ import annotations

import numpy as np
import pandas as pd

import pytest
from sklearn.metrics import r2_score, f1_score
from geml.classifiers import (
    GeneticProgrammingClassifier,
    HillClimbingClassifier,
    OnePlusOneClassifier,
    RandomSearchClassifier,
)
from geml.regressors import (
    GeneticProgrammingRegressor,
    HillClimbingRegressor,
    OnePlusOneRegressor,
    RandomSearchRegressor,
)


class TestSklearnAPI:

    @pytest.mark.parametrize(
        "classifier",
        [GeneticProgrammingClassifier, HillClimbingClassifier, RandomSearchClassifier, OnePlusOneClassifier],
    )
    def test_classifier(self, classifier):
        X = pd.DataFrame({"a": [3, 3, 2, 2], "b": [1, 0, 1, 0]})
        y = np.array([1, 1, 0, 0])

        X_test = pd.DataFrame({"a": [3, 2], "b": [5, 10]})
        y_test = np.array([1, 0])

        c = classifier(max_time=5, remove_time_overheads=False)
        c.fit(X, y)
        y_pred = c.predict(X_test)
        k = f1_score(y_test, y_pred)
        assert k >= 0
        assert k <= 1

    @pytest.mark.parametrize(
        "regressor",
        [GeneticProgrammingRegressor, HillClimbingRegressor, RandomSearchRegressor, OnePlusOneRegressor],
    )
    def test_regressor(self, regressor):
        X = pd.DataFrame({"a": [3.0, 3.0, 2.0, 2.0], "b": [1.0, 0.0, 1.0, 0.0]})
        y = np.array([2.0, 3.0, 1.0, 2.0])

        X_test = pd.DataFrame({"a": [3.0, 2.0], "b": [3.0, 2.0]})
        y_test = np.array([0.0, 0.0])

        c = regressor(max_time=5, remove_time_overheads=False)
        c.fit(X, y)
        y_pred = c.predict(X_test)
        k = r2_score(y_test, y_pred)
        assert k <= 1
