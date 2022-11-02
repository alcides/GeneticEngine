from __future__ import annotations

import time
from typing import Any

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.algorithms.gp.structure import StoppingCriterium
from geneticengine.algorithms.heuristics import Heuristics
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import FitnessType
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


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
        callbacks (List[Callback]): The callbacks to define what is done with the returned prints from the algorithm (default = []).
    """

    def __init__(
        self,
        representation: Representation[Any],
        problem: Problem,
        random_source: Source,
        population_size: int,
        initializer: PopulationInitializer,
        step: GeneticStep,
        stopping_criterium: StoppingCriterium,
        callbacks: list[Callback] = None,
    ):
        self.representation = representation
        self.problem = problem
        self.initializer = initializer
        self.population_size = population_size
        self.random_source = random_source
        self.step = step
        self.stopping_criterium = stopping_criterium
        self.callbacks = callbacks or []

    def evolve(self) -> tuple[Individual, FitnessType, Any]:
        """The main function of the GP object. This function runs the GP
        algorithm over the set number of generations, evolving better
        solutions.

        Returns a tuple with the following arguments:     individual
        (Individual): The fittest individual after the algorithm has
        finished.     fitness (float): The fitness of above individual.
        phenotype (Any): The phenotype of the best individual.
        """

        population = self.initializer.initialize(
            self.problem,
            self.representation,
            self.random_source,
            self.population_size,
        )

        generation = 0
        start = time.time()
        elapsed_time = lambda: time.time() - start

        while not self.stopping_criterium.is_ended(
            population,
            generation,
            elapsed_time(),
        ):
            generation += 1
            population = self.step.iterate(
                self.problem,
                self.representation,
                self.random_source,
                population,
                self.population_size,
            )
            for cb in self.callbacks:
                cb.process_iteration(generation, population, elapsed_time(), gp=self)

        self.final_population = population
        best_individual = self.get_best_individual(self.problem, population)

        for cb in self.callbacks:
            cb.end_evolution()
        return (
            best_individual,
            self.evaluate(best_individual),
            self.representation.genotype_to_phenotype(
                best_individual.genotype,
            ),
        )
