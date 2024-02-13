from __future__ import annotations

import pandas as pd

from sklearn.metrics import accuracy_score
from geml.classifiers import GeneticProgrammingClassifier


class TestSklearnAPI:
    def test_classifier(self):
        X = pd.DataFrame({"a": [3, 3, 2, 2], "b": [1, 0, 1, 0]})
        y = [1, 1, 0, 0]

        X_test = pd.DataFrame({"a": [3, 2], "b": [5, 10]})
        y_test = [1, 0]

        c = GeneticProgrammingClassifier(scoring=accuracy_score, max_depth=7)
        c.fit(X, y)
        y_pred = c.predict(X_test)
        k = accuracy_score(y_test, y_pred)
        assert k >= 0
        assert k <= 1
