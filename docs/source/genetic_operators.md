# Genetic operators

## Population Initializer

### Full population initialization

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.initializers.FullInitializer
```

### Grow population initialization

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.initializers.GrowInitializer
```

### Combine Initializers

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.initializers.HalfAndHalfInitializer
```

### Ramped Half-and-Half population initialization

This option is only available for the tree-based representation. Although the same approach can be used in Grammatical Evolution-based approaches, by constraining the maximum depth in generation 0 to 3, and allowing it to go to depth 10 in generation 2, it changes the genotype-to-mapping function so that the same genotype will lead to a different phenotype, just because the maximum allowed depth changed, not because of any genetic operator.

```{eval-rst}
.. autoapiclass:: geneticengine.core.representations.tree.operators.RampedHalfAndHalfInitializer
```


### How to inject pre-existing programs into the initial population.

The `InjectInitialPopulationWrapper` class allows you to pass a list of programs to include in the initial population. The remainder of the initial population will be selected via a backup initializer.

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.initializers.InjectInitialPopulationWrapper
```

## Selection operators

### Tournament Selection

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.selection.TournamentSelection
```

### Lexicase Selection

This function generates a selection operator that uses Lexicase Selection [^1]

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.selection.LexicaseSelection
```

## Elitism and Novelty

### Elitism

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.elitism.ElitismStep
```

### Novelty

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.novelty.NoveltyStep
```

## Mutation and Crossover

### Mutation

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.mutation.GenericMutationStep
```

Note that the operator parameter allows different representations to introduce their own custom mutators.

### Crossover

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.crossover.GenericCrossoverStep
```

Note that the operator parameter allows different representations to introduce their own custom crossover operators.

## Combinators

Combinators allow you to design your own evolutionary algorithms. In the following example, we are defining a GP variant that applies a tournament selection of size 5, and then takes the best 5% of the population, merges with new elements that will make up 5% of the new population, and the remaining 90% will be the result of a crossover with 0.01% of probability, after which a mutation with 90% of probability is applied.

```
default_generic_programming_step = ParallelStep([
        ElitismStep(),
        NoveltyStep()
        SequenceStep(
            TournamentSelection(5),
            GenericCrossoverStep(0.01),
            GenericMutationStep(0.9),
        )
    ], weights=[5, 5, 90]),),


)
```

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.combinators.SequenceStep
```

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.crossover.ParallelStep
```

##### References


[^1]: T. Helmuth, L. Spector and J. Matheson, "Solving Uncompromising Problems With Lexicase Selection," in IEEE Transactions on Evolutionary Computation, vol. 19, no. 5, pp. 630-643, Oct. 2015.
