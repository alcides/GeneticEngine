# Sklearn-compatible API

If you want to use Genetic Engine in your Machine Learning pipelines, you can use it, through these examples:

## Classification

```
from geneticengine.off_the_shelf.classifiers import GeneticProgrammingClassifier

model = GeneticProgrammingClassifier()
model.fit(X, y)
```

```{eval-rst}
.. autoapiclass:: geneticengine.off_the_shelf.classifiers.GeneticProgrammingClassifier
```

## Regression

```
from geneticengine.off_the_shelf.regressors import GeneticProgrammingRegressor

model = GeneticProgrammingRegressor()
model.fit(X, y)
```

```{eval-rst}
.. autoapiclass:: geneticengine.off_the_shelf.regressors.GeneticProgrammingRegressor
```


For complete examples, check the `sklearn-type-examples.py` in the examples folder.
