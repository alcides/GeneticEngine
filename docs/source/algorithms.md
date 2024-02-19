# Algorithms

Genetic Engine suports a list of different algorithms:


## Genetic Programming

Genetic Programming is supported via two interfaces. `SimpleGP` allows you defined all paramaters without creating objects or importing functions.
It is easier to use, if you are looking for a standard GP implementation.

```{eval-rst}
.. autoapiclass:: geml.simplegp.SimpleGP
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

## 1+1 Evolutionary Algorithm

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.one_plus_one.OnePlusOne
```

## MultiPopulation Genetic Programming

This is a version of Genetic Programming, which has multiple populations that work independently, even with their own Problem instances.

There is a new, optional migration step (and migration_size), that selects individuals from other populations, to allow some transference of individuals from one population to the other.

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.MultiPopulationGP
```
