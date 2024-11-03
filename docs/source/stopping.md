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

## Budget Combinators

## AnyOf

Terminates when either of the two criteria is true.

```{eval-rst}
.. autoapiclass:: geneticengine.evaluation.budget.AnyOf
```
