# Sklearn-compatible API

If you want to use Genetic Engine in your Machine Learning pipelines, you can use it, through these examples:

## Classification

```
from geml.classifiers import GeneticProgrammingClassifier

model = GeneticProgrammingClassifier()
model.fit(X, y)
```

```{eval-rst}
.. autoapiclass:: geml.classifiers.GeneticProgrammingClassifier
```

## Regression

```
from geml.regressors import GeneticProgrammingRegressor

model = GeneticProgrammingRegressor()
model.fit(X, y)
```

```{eval-rst}
.. autoapiclass:: geml.regressors.GeneticProgrammingRegressor
```


For complete examples, check the `sklearn-type-examples.py` in the examples folder.
