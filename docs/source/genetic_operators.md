# Genetic operators

## Starndard Population Initializers

These are two population initializers that work across all representations:

### StandardInitializer

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.initializers.StandardInitializer
```

### HalfAndHalfInitializer

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.initializers.HalfAndHalfInitializer
```

## Tree-based Population Initializers

These initializers are only available for the tree-based representation.

 Although the same approach can be used in Grammatical Evolution-based approaches, it would be too artificial.

 If in the initial population, we define a maximum-depth of 5 (e.g., forced by the grow method), by the time you get to the first generation with a maximum allowed of 10, the genotype-to-phenotype mapping will ignore the structure defined by grow, and will create a completely different individual, even though the genotype did not change since the initial population.

### Full population initialization

```{eval-rst}
.. autoapiclass:: geneticengine.representations.tree.operators.FullInitializer
```

### Grow population initialization

```{eval-rst}
.. autoapiclass:: geneticengine.representations.tree.operators.GrowInitializer
```

### Position Independent Grow population initialization

```{eval-rst}
.. autoapiclass:: geneticengine.representations.tree.operators.PositionIndependentGrowInitializer
```

### Ramped Half-and-Half population initialization



```{eval-rst}
.. autoapiclass:: geneticengine.representations.tree.operators.RampedHalfAndHalfInitializer
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

Combinators allow you to design your own evolutionary algorithms. When they are created, they are agnostic to the population size, so they can be used inside other combinator steps (like SequenceStep or ParallelStep).


In the following example, we are defining a Genetic Programming step that can be used in the Genetic Programming algorithm.

```
default_generic_programming_step = ParallelStep([
        ElitismStep(),
        NoveltyStep()
        SequenceStep(
            TournamentSelection(tournament_size=5),
            GenericCrossoverStep(probability=0.01),
            GenericMutationStep(probability=0.9),
        )
    ], weights=[.05, .05, .90])
)
```

In that case, and according to the ParallelStep, a new generation with population size 1000 will have:

1. the best 50 individuals of the previous generation (5% of elistim);
2. 50 new individuals (5% of novelty); and
3. 900 individuals that are create as follows (90% according to the Sequence Step weight):

    * First, 900 individuals will be selected using a tournament each. For each tournament, 5 individuals are chosen at random from the original population, and the best will pass to the next step.
    * The 900 individuals selected via tournament have a 1% probability of being crossover with another individual. Otherwise they will remain to the next step.
    * The 900 individuals that went through crossover or not are not mutated with 90% of probability.


```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.combinators.SequenceStep
```


```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.combinators.ParallelStep
```

```{eval-rst}
.. autoapiclass:: geneticengine.algorithms.gp.operators.combinators.ExclusiveParallelStep
```

##### References


[^1]: T. Helmuth, L. Spector and J. Matheson, "Solving Uncompromising Problems With Lexicase Selection," in IEEE Transactions on Evolutionary Computation, vol. 19, no. 5, pp. 630-643, Oct. 2015.
