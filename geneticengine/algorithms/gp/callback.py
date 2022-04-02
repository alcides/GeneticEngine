from __future__ import annotations

from abc import ABC


class Callback(ABC):
    def process_iteration(self, generation: int, population, time: float, gp):
        ...

    def end_evolution(self):
        ...


class DebugCallback(Callback):
    """Example of a callback that prints all the individuals in the population"""

    def process_iteration(self, generation: int, population, time: float, gp):
        for p in population:
            print(p)
        print(f"___ end of gen {generation}")


class ProgressCallback(Callback):
    """Prints the number of the generation"""

    def process_iteration(self, generation: int, population, time: float, gp):
        print(f"Generation {generation}. Time {time}")


class PrintBestCallback(Callback):
    """Prints the number of the generation"""

    def process_iteration(self, generation: int, population, time: float, gp):
        fitness = round(gp.evaluate(population[0]), 4)
        if not gp.timer_stop_criteria:
            print(f"Generation {generation} / {gp.number_of_generations}. Time {time}")
        else:
            print(f"Generation {generation}. Time {time} / {gp.timer_stop_criteria}")
        print(f"Best fitness: {fitness}")
        print(f"Best genotype: {population[0].genotype}")
