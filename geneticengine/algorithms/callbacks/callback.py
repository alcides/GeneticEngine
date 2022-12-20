from __future__ import annotations

from abc import ABC

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.algorithms.gp.operators.stop import TimeStoppingCriterium
from geneticengine.core.problems import FitnessType
from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.utils import average_fitness


class Callback(ABC):
    def process_iteration(self, generation: int, population, time: float, gp):
        ...

    def end_evolution(self):
        ...


def pretty_print_fitness(best_individual: Individual, gp) -> str:
    fitness: FitnessType
    if isinstance(gp.problem, SingleObjectiveProblem):
        fitness = best_individual.evaluate(gp.problem)

    elif isinstance(gp.problem, MultiObjectiveProblem):
        if gp.problem.number_of_objectives() > 10:
            if best_individual.fitness == None:
                best_individual.evaluate(problem=gp.problem)
            fitness = average_fitness(best_individual)
        else:
            fitness = [fitness for fitness in gp.problem.evaluate(best_individual.get_phenotype())]

    else:
        assert False

    if isinstance(fitness, float):
        return f"{fitness:.4f}"
    else:
        return ", ".join([f"{component:.4f}" for component in fitness])


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
        best_fitness = pretty_print_fitness(best_individual, gp)

        print(
            f"[{self.__class__.__name__}] Generation {generation}. Time {time:.2f}. Best fitness: {best_fitness}",
        )


class PrintBestCallback(Callback):
    """Prints the number of the generation."""

    def process_iteration(self, generation: int, population, time: float, gp):
        fitness: FitnessType

        best_individual: Individual = gp.get_best_individual(gp.problem, population)
        best_fitness = pretty_print_fitness(best_individual, gp)

        if isinstance(gp.stopping_criterium, TimeStoppingCriterium):
            print(
                f"[{self.__class__.__name__}] Generation {generation}. Time {time:.2f} / {gp.stopping_criterium.max_time}",
            )
        elif isinstance(gp.stopping_criterium, GenerationStoppingCriterium):
            print(
                f"[{self.__class__.__name__}] Generation {generation} / {gp.stopping_criterium.max_generations}. Time {time:.2f}",
            )
        print(f"[{self.__class__.__name__}] Best fitness: {best_fitness}")
        print(f"[{self.__class__.__name__}] Best genotype: {best_individual}")
