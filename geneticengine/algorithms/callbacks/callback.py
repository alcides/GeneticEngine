from __future__ import annotations

from abc import ABC
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium, TimeStoppingCriterium

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
            fitness = round(gp.problem.evaluate(best_individual.get_phenotype()), 4)

        elif isinstance(gp.problem, MultiObjectiveProblem):
            fitness = [round(fitness, 4) for fitness in gp.problem.evaluate(best_individual.get_phenotype())]
            if len(fitness) > 10:
                fitness = round(average_fitness(best_individual), 4)

        print(
            f"[{self.__class__}] Generation {generation}. Time {round(time, 2)}. Best fitness: {fitness}",
        )


class PrintBestCallback(Callback):
    """Prints the number of the generation"""

    def process_iteration(self, generation: int, population, time: float, gp):

        best_individual = gp.get_best_individual(gp.problem, population)

        if isinstance(gp.problem, SingleObjectiveProblem):
            fitness = round(gp.problem.evaluate(best_individual.get_phenotype()), 4)
        elif isinstance(gp.problem, MultiObjectiveProblem):
            fitness = [round(fitness, 4) for fitness in gp.problem.evaluate(best_individual.get_phenotype())]
            if len(fitness) > 10:
                fitness = sum(fitness) / len(fitness)

        if isinstance(gp.stopping_criterium, TimeStoppingCriterium):
            print(
                f"[{self.__class__}] Generation {generation}. Time {round(time, 2)} / {gp.stopping_criterium.max_time}",
            )
        elif isinstance(gp.stopping_criterium, GenerationStoppingCriterium):
            print(
                f"[{self.__class__}] Generation {generation} / {gp.stopping_criterium.max_generations}. Time {time}",
            )
        print(f"[{self.__class__}] Best fitness: {fitness}")
        print(f"[{self.__class__}] Best genotype: {population[0].genotype}")
