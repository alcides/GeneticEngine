# Genetic operators

## Population Initializer

### Full population initialization

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.generation_steps.initializers.FullInitializer
```

### Grow population initialization

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.generation_steps.initializers.GrowInitializer
```

### Ramped Half-and-Half population initialization

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.core.representations.tree.operators.RampedHalfAndHalfInitializer
```


### How to inject pre-existing programs into the initial population.

The `InjectInitialPopulationWrapper` class allows you to pass a list of programs to include in the initial population. The remainder of the initial population will be selected via a backup initializer.

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.generation_steps.initializers.InjectInitialPopulationWrapper
```

## Selection operators

### Tournament Selection

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.generation_steps.selection.TournamentSelection
```

### Lexicase Selection

This function generates a selection operator that uses Lexicase Selection [^1]

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.generation_steps.selection.LexicaseSelection
```

## Elitism and Novelty

### Elitism

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.generation_steps.elitism.ElitismStep
```

### Novelty

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.generation_steps.novelty.NoveltyStep
```

## Mutation and Crossover

### Mutation

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.generation_steps.mutation.GenericMutationStep
```

### Crossover

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.generation_steps.crossover.GenericCrossoverStep
```

## Combinators

Combinators allow you to design your own evolutionary algorithms. In the following example, we are defining a GP variant that applies a tournament selection of size 5, and then takes the best 5% of the population, merges with new elements that will make up 5% of the new population, and the remaining 90% will be the result of a crossover with 0.01% of probability, after which a mutation with 90% of probability is applied.

```
default_generic_programming_step = SequenceStep(
    TournamentSelection(5),
    ParallelStep([
        SequenceStep(GenericCrossoverStep(0.01), GenericMutationStep(0.9)),
        ElitismStep(),
        NoveltyStep()
    ], weights=[90, 5, 5]),
)
```


##### References


[^1]: T. Helmuth, L. Spector and J. Matheson, "Solving Uncompromising Problems With Lexicase Selection," in IEEE Transactions on Evolutionary Computation, vol. 19, no. 5, pp. 630-643, Oct. 2015.
