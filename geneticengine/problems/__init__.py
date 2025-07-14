from __future__ import annotations

import abc
import sys
from typing import Callable, Generic, NamedTuple, Optional
from typing import TypeVar

class InvalidFitnessException(Exception):
    pass

class Fitness(NamedTuple):
    fitness_components: list[float]
    valid : bool = True

    def __str__(self):
        if self.valid:
            m = ",".join([  str(x) for x in self.fitness_components])
            return f"[{m}]"
        else:
            return "<invalid_fitness>"



P = TypeVar("P")


class Problem(abc.ABC, Generic[P]):
    """Represents the Optimization Problem being solved."""

    minimize: list[bool]
    target: Optional[list[float]]

    def __init__(
        self,
        fitness_function: Callable[[P], list[float]],
        minimize: list[bool],
        target: Optional[list[float]] = None,
    ):
        self.minimize = minimize
        self.target = target
        self.ff = {"ff": fitness_function}

    def evaluate(self, phenotype: P) -> Fitness:
        v = self.ff["ff"](phenotype)
        return Fitness(v)

    @abc.abstractmethod
    def is_better(self, a: Fitness, b: Fitness) -> bool:
        """Returns whether the first fitness is better than the second."""
        ...

    def get_invalid_fitness(self) -> Fitness:
        """Returns an invalid fitness."""
        return Fitness([ -(sys.maxsize -1) if m else sys.maxsize for m in self.minimize], valid=False)

    def number_of_objectives(self) -> int:
        return len(self.minimize)

    def is_solved(self, fitness: Fitness) -> bool:
        if self.target is None:
            return False
        elif not fitness.valid:
            return False
        else:
            return all(
                a <= t if mi else a >= t for (a, t, mi) in zip(fitness.fitness_components, self.target, self.minimize)
            )


class SequentialObjectiveProblem(Problem[P]):
    """SequentialObjectiveProblem is defined by a list of objectives that are intended to be either maximized/minimized in order."""

    def is_better(self, a: Fitness, b: Fitness) -> bool:
        if not a.valid:
            return False
        if not b.valid:
            return True

        for af, bf, m in zip(a.fitness_components, b.fitness_components, self.minimize):
            if m and af > bf:
                return False
            elif not m and af < bf:
                return False
        return True


class SingleObjectiveProblem(SequentialObjectiveProblem[P]):
    """A problem that is characterized by a single value."""

    def __init__(self, fitness_function: Callable[[P], float], minimize: bool = False, target: Optional[float] = None):
        super().__init__(lambda x: [fitness_function(x)], [minimize], None if target is None else [target])


class MultiObjectiveProblem(Problem[P]):
    def is_better(self, a: Fitness, b: Fitness) -> bool:
        if not a.valid:
            return False
        if not b.valid:
            return True

        """To be better in a multi-objective setting, it needs to be better in all objectives."""
        return all(
            a <= t if mi else a >= t for (a, t, mi) in zip(a.fitness_components, b.fitness_components, self.minimize)
        )


class LazyMultiObjectiveProblem(MultiObjectiveProblem):
    """LazyMultiObjectiveProblem is used for problems whose number of objectives is not known a-priori."""

    def __init__(
        self,
        fitness_function: Callable[[P], list[float]],
        minimize: bool = False,
        target: Optional[float] = None,
    ):
        self.initialized = False
        self.future_minimize = minimize
        self.future_target = target
        super().__init__(fitness_function, None, None)

    def evaluate(self, phenotype: object) -> Fitness:
        v = self.ff["ff"](phenotype)
        if not self.initialized:
            self.minimize = [self.future_minimize for _ in v]
            if self.future_target is not None:
                self.target = [self.future_target for _ in v]
            self.initialized = True
        return Fitness(v)

    def number_of_objectives(self) -> int:
        assert self.minimize is not None, "Please evaluate an individual before consulting the number of objectives."
        return super().number_of_objectives()


__ALL__ = [
    Fitness,
    Problem,
    SingleObjectiveProblem,
    SequentialObjectiveProblem,
    MultiObjectiveProblem,
    LazyMultiObjectiveProblem,
]
