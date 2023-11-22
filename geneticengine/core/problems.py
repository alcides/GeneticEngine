from __future__ import annotations

import abc
from typing import Callable, NamedTuple
from typing import TypeVar, Any


class Fitness(NamedTuple):
    maximizing_aggregate: float
    fitness_components: list[float]

    def __str__(self):
        return "|".join([f"{d:.5f}" for d in self.fitness_components])


P = TypeVar("P")


class Problem(abc.ABC):
    """An Abstract class that SingleObjectiveProblem and MultiObjectiveProblem
    extends to.

    Args:
        minimize (bool | list[bool]): When switch on, the fitness function is reversed, so that a higher result from the
            fitness function corresponds to a less fit solution.
        fitness_function (Callable[[P], float] | Callable[[P], list[float]]): The fitness function. Should take in any
            valid individual and return a float or a list of floats, depending if its a single objetive problem or a
            multi objective problem.
    """

    minimize: list[bool] | bool
    epsilon: float

    @abc.abstractmethod
    def evaluate(self, phenotype: P) -> Fitness:
        ...

    def key_function(self, a: Fitness) -> float:
        """Returns the (maximizing) fitness of the individual as a single
        float."""
        return a.maximizing_aggregate

    def is_better(self, a: Fitness, b: Fitness) -> bool:
        """Returns whether the first fitness is better than the second."""
        return a.maximizing_aggregate > b.maximizing_aggregate

    @abc.abstractmethod
    def number_of_objectives(self) -> int:
        ...


class SingleObjectiveProblem(Problem):
    """SingleObjectiveProblem is a class that extends the Problem class.

    Args:
        minimize (bool): When switch on, the fitness function is reversed, so that a higher result from the fitness
            function corresponds to a less fit solution.
        fitness_function (Callable[[P], float]): The fitness function. Should take in any valid individual and return a
            float.
    """

    # Uses dict to avoid the mismatch between functions and methods (first argument)

    def __init__(
        self,
        fitness_function: Callable[[P], float],
        minimize: bool = False,
    ):
        self.ff = {"ff": fitness_function}
        self.minimize = [minimize]

    def evaluate(self, phenotype: P) -> Fitness:
        v = float(self.ff["ff"](phenotype))
        minimize_value = self.minimize[0] if isinstance(self.minimize, list) else self.minimize
        key = -v if minimize_value else v
        return Fitness(key, [v])

    def number_of_objectives(self) -> int:
        return 1


class MultiObjectiveProblem(Problem):
    """MultiObjectiveProblem is a class that extends the Problem class.

    Args:
        minimize (list[bool] | bool): When switch on, the fitness function is reversed, so that a higher result from the
            fitness function corresponds to a less fit solution.
            When a list is passed, each element of the list corresponds to a fitness component. When a bool is passed
            all the fitness component of the problem are minimized or maximized. when giving a bool, the selection
            algorithm must have that into consideration and create a list of bools with the same size as the number of
            the fitness components of an Individual.
        fitness_function (Callable[[P], list[bool]]): The fitness function. Should take in any valid individual and
            return a list of float.
        best_individual_criteria_function (Optional(Callable[[P], float]): This function allow the user to choose how to
            find the best individual in a generation (default = None , this means that the individual with the best
            fitness is the one considered as the best in that generation)
    """

    minimize: list[bool] | bool
    ff: dict[str, Any]

    def __init__(
        self,
        minimize: list[bool] | bool,
        fitness_function: Callable[[P], list[float]],
        best_individual_criteria_function: Callable[[P], float] | None = None,
        aggregate_fitness: Callable[[list[float]], float] | None = None,
    ):
        self.minimize = minimize

        def default_single_objective_merge(d: Any) -> float:
            if isinstance(self.minimize, list):
                return sum(m and -fit or +fit for (fit, m) in zip(fitness_function(d), self.minimize))
            elif isinstance(self.minimize, bool):
                return sum(-fit if self.minimize else fit for fit in fitness_function(d))
            else:
                assert False, "minimize must be either a list[bool] or a bool"



        self.ff = {
            "ff": fitness_function,
            "best_individual": best_individual_criteria_function or default_single_objective_merge,
            "aggregate_fitness": aggregate_fitness,
        }

    def evaluate(self, phenotype: P) -> Fitness:
        lst: list[float] = self.ff["ff"](phenotype)
        multiple = [float(x) for x in lst]
        if self.ff["aggregate_fitness"] is None:
            single = self.ff["best_individual"](phenotype)
        else:
            single = self.ff["aggregate_fitness"](multiple)
        return Fitness(single, multiple)

    def number_of_objectives(self) -> int:
        return len(self.minimize) if isinstance(self.minimize, list) else 1


def wrap_depth_minimization(p: SingleObjectiveProblem) -> SingleObjectiveProblem:
    """This wrapper takes a SingleObjectiveProblem and adds a penalty for
    bigger trees."""

    def w(i):
        if p.minimize:
            return p.evaluate(i)[0] + i.gengy_distance_to_term * 10**-25
        else:
            return p.evaluate(i)[0] - i.gengy_distance_to_term * 10**-25

    return SingleObjectiveProblem(
        minimize=p.minimize if isinstance(p.minimize, bool) else p.minimize[0],
        fitness_function=w
    )
