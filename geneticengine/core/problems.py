from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Generic
from typing import TypeVar, Any

from geneticengine.core.generic_utils import GenericWrapper


class Fitness(abc.ABC):
    pass


@dataclass
class FitnessSingleObjective(Fitness):
    """Represents the fitness of an individual in a SingleObjectiveProblem."""

    fitness: float

    def __str__(self):
        return f"{self.fitness:.4f}"


@dataclass
class FitnessMultiObjective(Fitness):
    """Represents the fitness of an individual in a MultiObjectiveProblem."""

    multiple_fitnesses: list[float]
    fitness: float

    def __str__(self):
        return ", ".join([f"{component:.4f}" for component in self.multiple_fitnesses]) + " ({self.fitness:.4f})"


P = TypeVar("P")
SingleObjectiveCallable = Callable[[P], float]
MultiObjectiveCallable = Callable[[P], list[float]]


FT = TypeVar("FT")


class Problem(abc.ABC, Generic[FT]):
    """An Abstract class that SingleObjectiveProblem and MultiObjectiveProblem
    extends to.

    Args:
        minimize (bool | list[bool]): When switch on, the fitness function is reversed, so that a higher result from the
            fitness function corresponds to a less fit solution.
        fitness_function (Callable[[P], float] | Callable[[P], list[float]]): The fitness function. Should take in any
            valid individual and return a float or a list of floats, depending if its a single objetive problem or a
            multi objective problem.
    """

    minimize: bool | list[bool]
    fitness_function: SingleObjectiveCallable | MultiObjectiveCallable

    @abc.abstractmethod
    def evaluate(self, phenotype: P) -> FT:
        ...

    @abc.abstractmethod
    def key_function(self, a: P) -> float:
        """Returns the (maximizing) fitness of the individual as a single
        float."""
        ...

    @abc.abstractmethod
    def is_better(self, a: FT, b: FT) -> bool:
        """Returns whether the first fitness is better than the second."""
        ...


class SingleObjectiveProblem(Problem[FitnessSingleObjective]):
    """SingleObjectiveProblem is a class that extends the Problem class.

    Args:
        minimize (bool): When switch on, the fitness function is reversed, so that a higher result from the fitness
            function corresponds to a less fit solution.
        fitness_function (Callable[[P], float]): The fitness function. Should take in any valid individual and return a
            float.
    """

    # Uses dict to avoid the mismatch between functions and methods (first argument)
    fitness_function_host: GenericWrapper[SingleObjectiveCallable]
    minimize: bool

    def __init__(self, fitness_function: SingleObjectiveCallable, minimize: bool = False):
        self.fitness_function_host = GenericWrapper(fitness_function)
        self.minimize = minimize

    def evaluate(self, phenotype: P) -> FitnessSingleObjective:
        c: SingleObjectiveCallable = self.fitness_function_host.get()
        v = float(c(phenotype))
        return FitnessSingleObjective(fitness=v)

    def key_function(self, a: P) -> float:
        if self.minimize:
            return -self.evaluate(a).fitness
        else:
            return self.evaluate(a).fitness

    def is_better(self, a: FT, b: FT) -> bool:
        assert isinstance(a, FitnessSingleObjective)
        assert isinstance(b, FitnessSingleObjective)
        if self.minimize:
            return a.fitness < b.fitness
        else:
            return a.fitness > b.fitness


class MultiObjectiveProblem(Problem[FitnessMultiObjective]):
    """MultiObjectiveProblem is a class that extends the Problem class.

    Args:
        minimize (list[bool]): When switch on, the fitness function is reversed, so that a higher result from the
            fitness function corresponds to a less fit solution.
        fitness_function (Callable[[P], list[bool]]): The fitness function. Should take in any valid individual and
            return a list of float.
        best_individual_criteria_function (Optional(Callable[[P], float]): This function allow the user to choose how to
            find the best individual in a generation (default = None , this means that the individual with the best
            fitness is the one considered as the best in that generation)
    """

    minimize: list[bool]
    fitness_function_host: GenericWrapper[MultiObjectiveCallable]
    best_individual_criteria_function: GenericWrapper[SingleObjectiveCallable]

    def __init__(
        self,
        minimize: list[bool],
        fitness_function: MultiObjectiveCallable,
        best_individual_criteria_function: SingleObjectiveCallable | None,
    ):
        self.minimize = minimize
        self.fitness_function_host = GenericWrapper(fitness_function)

        def default_single_objective_merge(d: Any) -> float:
            return sum(m and -fit or +fit for (fit, m) in zip(fitness_function(d), self.minimize))

        self.best_individual_criteria_function = GenericWrapper(
            best_individual_criteria_function or default_single_objective_merge,
        )

    def number_of_objectives(self):
        return len(self.minimize)

    def evaluate(self, phenotype: P) -> FitnessMultiObjective:
        multiple = [float(x) for x in self.fitness_function_host.get()(phenotype)]
        single = self.best_individual_criteria_function.get()(phenotype)
        if single is None:
            single = sum(m and -fit or +fit for (fit, m) in zip(multiple, self.minimize))

        return FitnessMultiObjective(
            multiple_fitnesses=multiple,
            fitness=single,
        )

    def key_function(self, a: P) -> float:
        return self.evaluate(a).fitness

    def is_better(self, a: FT, b: FT) -> bool:
        assert isinstance(a, FitnessMultiObjective)
        assert isinstance(b, FitnessMultiObjective)
        if self.minimize:
            return a.fitness < b.fitness
        else:
            return a.fitness > b.fitness


def wrap_depth_minimization(p: SingleObjectiveProblem) -> SingleObjectiveProblem:
    """This wrapper takes a SingleObjectiveProblem and adds a penalty for
    bigger trees."""

    def w(i):
        if p.minimize:
            return p.evaluate(i).fitness + i.gengy_distance_to_term * 10**-25
        else:
            return p.evaluate(i).fitness - i.gengy_distance_to_term * 10**-25

    return SingleObjectiveProblem(minimize=p.minimize, fitness_function=w)
