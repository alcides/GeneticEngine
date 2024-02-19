from __future__ import annotations
from functools import reduce


from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.initializers import (
    StandardInitializer,
)
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.algorithms.heuristics import HeuristicSearch
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource

from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.tracker import SingleObjectiveProgressTracker
from geneticengine.representations.api import Representation


def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


def default_multipopulation_step():
    """The default step in Genetic Programming."""
    return ParallelStep(
        [
            ElitismStep(),
            NoveltyStep(),
            SequenceStep(
                TournamentSelection(5),
                GenericCrossoverStep(0.01),
                GenericMutationStep(0.9),
            ),
        ],
        weights=[5, 5, 90],
    )


class MultiPopulationGP(HeuristicSearch):
    """MultiPopulation version of Genetic Programming.

    Populations evolve independently if migration_size = 0. Otherwise, that number of individuals is selected from the other populations, according to a migration step (a Tournament by default).

    Args:
        representation (Representation): The individual representation used by the GP program.
        problem (Problem): A SingleObjectiveProblem or a MultiObjectiveProblem problem.
        random (RandomSource): A RNG instance
        population_sizes (list[int]): The size of each population (default = [50, 50, 50, 50]).
        initializer (PopulationInitializer): The method to generate new individuals.
        step (GeneticStep): The main step of evolution.
        budget (SearchBudget): How much effort can be used in the search.
        callbacks (List[Callback]): The callbacks to define what is done with the returned prints from the algorithm
            (default = []).
    """

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: Representation,
        random: RandomSource = None,
        recorder: SingleObjectiveProgressTracker | None = None,
        population_sizes: list[int] = [50, 50, 50, 50],
        population_initializers: list[PopulationInitializer] = None,
        step: GeneticStep | None = None,
        migration_step: GeneticStep = TournamentSelection(2),
        migration_size: int = 0,
    ):
        super().__init__(problem, budget, representation, random, recorder)
        self.population_sizes = population_sizes
        self.population_initializers = (
            population_initializers
            if population_initializers is not None
            else [StandardInitializer() for _ in population_sizes]
        )
        self.step = step if step is not None else default_multipopulation_step()
        self.migration_step = migration_step
        self.migration_size = migration_size
        if self.migration_size > 0:
            assert self.migration_step is not None

    def search(self) -> Individual:
        """The main function of the GP object. This function runs the GP
        algorithm over the set number of generations, evolving better
        solutions.

        Returns:
            A tuple with the individual, its fitness and its phenotype.
        """
        generation = 0
        populations = [
            initializer.initialize(
                self.problem,
                self.representation,
                self.random,
                pop_size,
            )
            for pop_size, initializer in zip(self.population_sizes, self.population_initializers)
        ]

        self.tracker.evaluate(list(flatten(populations)))

        while not self.is_done():
            generation += 1

            populations = [
                self.step.iterate(
                    self.problem,
                    self.tracker.evaluator,
                    self.representation,
                    self.random,
                    population,
                    population_size - self.migration_size,
                    generation,
                )
                for population, population_size in zip(populations, self.population_sizes)
            ]
            self.tracker.evaluate(list(flatten(populations)))

            if self.migration_size > 0 and self.migration_step is not None:
                for i, pop in enumerate(populations):
                    total_population = reduce(lambda x, y: x + y, populations[:i] + populations[i + 1 :])
                    bests = self.migration_step.iterate(
                        self.problem,
                        self.tracker.evaluator,
                        self.representation,
                        self.random,
                        total_population,
                        self.migration_size,
                        generation=generation,
                    )
                    pop.extend(bests)
        return self.tracker.get_best_individual()
