from __future__ import annotations

from abc import ABC

from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.problems import SingleObjectiveProblem


class Callback(ABC):
    def process_iteration(self, generation: int, population, time: float, gp):
        ...

    def end_evolution(self):
        ...


class DebugCallback(Callback):
    """Example of a callback that prints all the individuals in the population"""

    def process_iteration(self, generation: int, population, time: float, gp):
        for p in population:
            print(f"[{self.__class__}] {p} at generation {generation}")


class ProgressCallback(Callback):
    """Prints the number of the generation"""

    def process_iteration(self, generation: int, population, time: float, gp):
        if isinstance(gp.problem, SingleObjectiveProblem):
            fitness = round(gp.evaluate(population[0]), 4)
            print(
                f"[{self.__class__}] Generation {generation}. Time {time}. Best fitness: {fitness}",
            )

        elif isinstance(gp.problem, MultiObjectiveProblem):
            list_of_fitness = [
                round(fitness, 4) for fitness in gp.evaluate(population[0])
            ]
            print(
                f"[{self.__class__}] Generation {generation}. Time {time}. List of fitness: {list_of_fitness}",
            )


class PrintBestCallback(Callback):
    """Prints the number of the generation"""

    def process_iteration(self, generation: int, population, time: float, gp):
        fitness = round(gp.evaluate(population[0]), 4)
        if not gp.timer_stop_criteria:
            print(
                f"[{self.__class__}] Generation {generation} / {gp.number_of_generations}. Time {time}",
            )
        else:
            print(
                f"[{self.__class__}] Generation {generation}. Time {time} / {gp.timer_stop_criteria}",
            )
        print(f"[{self.__class__}] Best fitness: {fitness}")
        print(f"[{self.__class__}] Best genotype: {population[0].genotype}")
