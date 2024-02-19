# Stopping Criteria

Search algorithms can be defined with different budgets (subclasses of `geneticengine.evaluation.budget.SearchBudget`)

## Time Budget

```{eval-rst}
.. autoapiclass:: geneticengine.evaluation.budget.TimeBudget
```

## Evaluation Budget

```{eval-rst}
.. autoapiclass:: geneticengine.evaluation.budget.EvaluationBudget
```

## Target Fitness

For single-objective:

```{eval-rst}
.. autoapiclass:: geneticengine.evaluation.budget.TargetFitness
```

For multi-objective:

```{eval-rst}
.. autoapiclass:: geneticengine.evaluation.budget.TargetMultiFitness
```

## Budget Combinators

## AnyOf

Terminates when either of the two criteria is true.

```{eval-rst}
.. autoapiclass:: geneticengine.evaluation.budget.AnyOf
```
