from __future__ import annotations

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


class SequenceStep(GeneticStep):
    """Applies multiple steps in order."""

    def __init__(self, *steps: GeneticStep):
        self.steps = steps

    def iterate(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        for step in self.steps:
            population = step.iterate(
                problem,
                representation,
                random_source,
                population,
                target_size,
                generation,
            )
            assert isinstance(population, list)
            assert len(population) == target_size
        return population

    def __str__(self):
        return ";".join([f"({x})" for x in self.steps])


class ParallelStep(GeneticStep):
    """Splits the size of the target population according to the weights, but
    all parallel steps have access to the whole population."""

    def __init__(
        self,
        steps: list[GeneticStep],
        weights: list[float] | None = None,
    ):
        self.steps = steps
        self.weights = weights or [1 for _ in steps]
        assert len(self.steps) == len(self.weights)

    def iterate(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        total = sum(self.weights)
        indices = [0] + self.cumsum(
            [int(round(w * len(population) / total, 0)) for w in self.weights],
        )
        ranges = list(zip(indices, indices[1:]))
        assert len(ranges) == len(self.steps)

        retlist = self.concat(
            [
                step.iterate(
                    problem,
                    representation,
                    random_source,
                    population,
                    end - start,
                    generation,
                )
                for ((start, end), step) in zip(ranges, self.steps)
            ],
        )
        assert len(retlist) == target_size
        return retlist

    def concat(self, ls):
        rl = []
        for l in ls:
            rl.extend(l)
        return rl

    def cumsum(self, l):
        v = 0
        nl = []
        for i in l:
            v = v + i
            nl.append(v)
        return nl

    def __str__(self):
        return "||".join([f"({x})" for x in self.steps])


class ExclusiveParallelStep(ParallelStep):
    """Splits the population according to weights, and applies a different step
    to each subset of the population."""

    def iterate(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        total = sum(self.weights)
        indices = [0] + self.cumsum(
            [int(round(w * len(population) / total, 0)) for w in self.weights],
        )
        ranges = list(zip(indices, indices[1:]))
        assert len(ranges) == len(self.steps)

        retlist = self.concat(
            [
                step.iterate(
                    problem,
                    representation,
                    random_source,
                    population[start:end],
                    end - start,
                    generation,
                )
                for ((start, end), step) in zip(ranges, self.steps)
            ],
        )
        assert len(retlist) == target_size
        return retlist
