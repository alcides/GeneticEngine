from __future__ import annotations

from abc import ABC

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.algorithms.gp.operators.stop import TimeStoppingCriterium


class Callback(ABC):
    def process_iteration(self, generation: int, population, time: float, gp):
        ...

    def end_evolution(self):
        pass


class DebugCallback(Callback):
    """Example of a callback that prints all the individuals in the
    population."""

    def process_iteration(self, generation: int, population, time: float, gp):
        for p in population:
            print(f"[{self.__class__.__name__}] {p} at generation {generation}")


class ProgressCallback(Callback):
    """Prints the number of the generation."""

    # Currently this only work with GP, doesnt work with Hill Climbing and Random Search
    def process_iteration(self, generation: int, population, time: float, gp):

        best_individual = gp.get_best_individual(gp.problem, population)
        best_fitness = best_individual.get_fitness(gp.problem)

        print(
            f"[{self.__class__.__name__}] Generation {generation}. Time {time:.2f}. Best fitness: {best_fitness}",
        )
        gp.evaluator.eval(gp.problem, population)


class PrintBestCallback(Callback):
    """Prints the number of the generation."""

    def process_iteration(self, generation: int, population, time: float, gp):

        best_individual: Individual = gp.get_best_individual(gp.problem, population)
        best_fitness = best_individual.get_fitness(gp.problem)

        if isinstance(gp.stopping_criterium, TimeStoppingCriterium):
            mt = gp.stopping_criterium.max_time
            print(
                f"[{self.__class__.__name__}] Generation {generation}. Time {time:.2f} / {mt}",
            )
        elif isinstance(gp.stopping_criterium, GenerationStoppingCriterium):
            mg = gp.stopping_criterium.max_generations
            print(
                f"[{self.__class__.__name__}] Generation {generation} / {mg}. Time {time:.2f}",
            )
        print(f"[{self.__class__.__name__}] Best fitness: {best_fitness}")
        print(f"[{self.__class__.__name__}] Best genotype: {best_individual}")
