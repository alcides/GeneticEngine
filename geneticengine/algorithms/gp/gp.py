from __future__ import annotations

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


def default_generic_programming_step():
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


class GP(Heuristics):
    """Represents the Genetic Programming algorithm. Defaults as given in A
    Field Guide to GP, p.17, by Poli and Mcphee:

    Args:
        representation (Representation): The individual representation used by the GP program.
        problem (Problem): A SingleObjectiveProblem or a MultiObjectiveProblem problem.
        random_source (RandomSource]): A RNG instance
        population_size (int): The population size (default = 200).
        initializer (PopulationInitializer): The method to generate new individuals.
        step (GeneticStep): The main step of evolution.
        stopping_criterium (StoppingCriterium): The class that defines how the evolution stops.
        callbacks (List[Callback]): The callbacks to define what is done with the returned prints from the algorithm
            (default = []).
    """

    def __init__(
        self,
        representation: Representation[Any, Any],
        problem: Problem,
        random_source: Source = RandomSource(0),
        population_size: int = 200,
        initializer: PopulationInitializer = StandardInitializer(),
        step: GeneticStep | None = None,
        stopping_criterium: StoppingCriterium = GenerationStoppingCriterium(100),
        callbacks: list[Callback] | None = None,
        evaluator: Callable[[], Evaluator] = SequentialEvaluator,
    ):
        super().__init__(representation, problem, evaluator())
        self.initializer = initializer
        self.population_size = population_size
        self.random_source = random_source
        self.step = step if step else default_generic_programming_step()
        self.stopping_criterium = stopping_criterium
        self.callbacks = callbacks if callbacks else []

    def evolve(self) -> Individual:
        """The main function of the GP object. This function runs the GP
        algorithm over the set number of generations, evolving better
        solutions.

        Returns:
            A tuple with the individual, its fitness and its phenotype.
        """

        population = self.initializer.initialize(
            self.problem,
            self.representation,
            self.random_source,
            self.population_size,
        )
        self.evaluator.eval(self.problem, population)

        generation = 0
        start = time.time()

        def elapsed_time():
            return time.time() - start

        for cb in self.callbacks:
            cb.process_iteration(generation, population, elapsed_time(), gp=self)

        while not self.stopping_criterium.is_ended(
            self.problem,
            population,
            generation,
            elapsed_time(),
            self.evaluator,
        ):
            generation += 1
            population = self.step.iterate(
                self.problem,
                self.evaluator,
                self.representation,
                self.random_source,
                population,
                self.population_size,
                generation,
            )
            self.evaluator.eval(self.problem, population)
            assert len(population) == self.population_size
            for cb in self.callbacks:
                cb.process_iteration(generation, population, elapsed_time(), gp=self)

        self.final_population = population
        best_individual = self.get_best_individual(self.problem, population)
        for cb in self.callbacks:
            cb.end_evolution()
        return best_individual
