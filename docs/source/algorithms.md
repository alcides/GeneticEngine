# Algorithms

Genetic Engine suports a list of different algorithms:


## Genetic Programming

Genetic Programming is supported via two interfaces. `GPFriendly` allows you defined all paramaters without creating objects or importing functions.
It is easier to use, if you are looking for a standard GP implementation.

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.gp_friendly.GPFriendly
```

However, if you are looking to implement your own algorithm, or variation of GP, the `GP` class is the most suitable.

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.gp.GP
```

Note that the `step` parameter allows the user to build their own evolutionary algorithm. See [the list of available genetic operators](genetic_operators.md).


## Hill Climbing


```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.hill_climbing.HC
```


## Random Mutations

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.random_mutations.RandomMutations
```

## Random Search

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.random_search.RandomSearch
```
