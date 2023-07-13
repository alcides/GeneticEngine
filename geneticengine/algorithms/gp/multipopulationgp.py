from __future__ import annotations
from functools import reduce

import time
from typing import Any, Callable

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.initializers import (
    StandardInitializer,
)
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.algorithms.gp.structure import StoppingCriterium
from geneticengine.algorithms.heuristics import Heuristics
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.evaluators import Evaluator, SequentialEvaluator


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


class MultiPopulationGP(Heuristics):
    """MultiPopulation version of Genetic Programming.

    Populations evolve independently if migration_size = 0. Otherwise, that number of individuals is selected from the other populations, according to a migration step (a Tournament by default).

    Args:
        representation (Representation): The individual representation used by the GP program.
        problem (Problem): A SingleObjectiveProblem or a MultiObjectiveProblem problem.
        random_source (RandomSource]): A RNG instance
        population_sizes (list[int]): The size of each population (default = [50, 50, 50, 50]).
        initializer (PopulationInitializer): The method to generate new individuals.
        step (GeneticStep): The main step of evolution.
        stopping_criterium (StoppingCriterium): The class that defines how the evolution stops.
        callbacks (List[Callback]): The callbacks to define what is done with the returned prints from the algorithm
            (default = []).
    """

    def __init__(
        self,
        representation: Representation[Any, Any],
        problem: Problem | None = None,
        problems: list[Problem] | None = None,
        random_source: Source = RandomSource(0),
        population_sizes: list[int] = [50, 50, 50, 50],
        initializer: PopulationInitializer = StandardInitializer(),
        step: GeneticStep | None = None,
        migration_step: GeneticStep = TournamentSelection(2),
        migration_size: int = 0,
        stopping_criterium: StoppingCriterium = GenerationStoppingCriterium(100),
        callbacks: list[Callback] | None = None,
        evaluator: Callable[[], Evaluator] = SequentialEvaluator,
    ):
        if problem is not None:
            self.problems = [problem for _ in population_sizes]
        elif problems is not None:
            assert len(problems) == len(population_sizes)
            self.problems = problems
        else:
            raise ValueError("You need to either set the problems or problem arguments.")

        super().__init__(representation, self.problems[0], evaluator())
        self.initializer = initializer
        self.population_sizes = population_sizes
        self.random_source = random_source
        self.step = step if step else default_multipopulation_step()
        self.migration_size = migration_size
        self.migration_step = migration_step
        if self.migration_size > 0:
            assert self.migration_step is not None
        self.stopping_criterium = stopping_criterium
        self.callbacks = callbacks if callbacks else []

    def evolve(self) -> list[Individual]:
        """The main function of the GP object. This function runs the GP
        algorithm over the set number of generations, evolving better
        solutions.

        Returns:
            A tuple with the individual, its fitness and its phenotype.
        """

        populations = [
            self.initializer.initialize(
                problem,
                self.representation,
                self.random_source,
                pop_size,
            )
            for pop_size, problem in zip(self.population_sizes, self.problems)
        ]
        for population, problem in zip(populations, self.problems):
            self.evaluator.eval(problem, population)

        generation = 0
        start = time.time()

        def elapsed_time():
            return time.time() - start

        for cb in self.callbacks:
            for population in populations:
                cb.process_iteration(generation, population, elapsed_time(), gp=self)

        while all(
            not self.stopping_criterium.is_ended(problem, population, generation, elapsed_time(), self.evaluator)
            for population, problem in zip(populations, self.problems)
        ):
            generation += 1
            populations = [
                self.step.iterate(
                    problem,
                    self.evaluator,
                    self.representation,
                    self.random_source,
                    population,
                    population_size - self.migration_size,
                    generation,
                )
                for population, problem, population_size in zip(populations, self.problems, self.population_sizes)
            ]

            if self.migration_size > 0 and self.migration_step is not None:
                for i, pop, prob in zip(range(len(self.problems)), populations, self.problems):
                    total_population = reduce(lambda x, y: x + y, populations[:i] + populations[i + 1 :])
                    bests = self.migration_step.iterate(
                        prob,
                        self.evaluator,
                        self.representation,
                        self.random_source,
                        total_population,
                        self.migration_size,
                        generation=generation,
                    )
                    pop.extend(bests)

            for population, problem in zip(populations, self.problems):
                self.evaluator.eval(problem, population)

            assert all(
                len(population) == population_size
                for population, population_size in zip(populations, self.population_sizes)
            )
            for cb in self.callbacks:
                for i, population in enumerate(populations):
                    cb.process_iteration(generation, population, elapsed_time(), gp=self)

        self.final_populations = populations
        best_individuals = [
            self.get_best_individual(problem, population) for population, problem in zip(populations, self.problems)
        ]
        for cb in self.callbacks:
            cb.end_evolution()
        return best_individuals
