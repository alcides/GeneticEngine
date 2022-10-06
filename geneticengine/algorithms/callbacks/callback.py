from __future__ import annotations

from abc import ABC

from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.utils import average_fitness


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

    # Currently this only work with GP, doesnt work with Hill Climbing and Random Search
    def process_iteration(self, generation: int, population, time: float, gp):

        best_individual = gp.get_best_individual(gp.problem, population)

        if isinstance(gp.problem, SingleObjectiveProblem):
            fitness = round(gp.evaluate(best_individual), 4)

        elif isinstance(gp.problem, MultiObjectiveProblem):
            fitness = [round(fitness, 4) for fitness in gp.evaluate(best_individual)]
            if len(fitness) > 10:
                fitness = round(average_fitness(best_individual), 4)

        print(
            f"[{self.__class__}] Generation {generation}. Time {time}. Best fitness: {fitness}",
        )


class PrintBestCallback(Callback):
    """Prints the number of the generation"""

    def process_iteration(self, generation: int, population, time: float, gp):
        if isinstance(gp.problem, SingleObjectiveProblem):
            fitness = round(gp.evaluate(population[0]), 4)
        elif isinstance(gp.problem, MultiObjectiveProblem):
            fitness = [round(fitness, 4) for fitness in gp.evaluate(population[0])]
            if len(fitness) > 10:
                fitness = sum(fitness) / len(fitness)

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
