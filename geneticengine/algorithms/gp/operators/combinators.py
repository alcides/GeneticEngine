from __future__ import annotations
from copy import deepcopy

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.evaluators import Evaluator


class SequenceStep(GeneticStep):
    """Applies multiple steps in order, passing the output population of one
    step to the input population of the next step."""

    def __init__(self, *steps: GeneticStep):
        self.steps = steps

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        for step in self.steps:
            population = step.iterate(
                problem,
                evaluator,
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
    """Creates a new population, using different steps for different slices of
    the target population. The input population for each parallel step is
    always the complete input population. The output/target population is the
    one that is split across all of the slices. The size of each slice is given
    by the proportion of the weight of that particular weight, compared to the
    sum of all weights.

    Consider the example:

    another_step = ParallelStep([AStep(), BStep()], weights=[2,3])

    In this example, the first 2/5 of the next population will be generated using AStep(), and the next 3/5 will be generated using BStep.
    """

    def __init__(
        self,
        steps: list[GeneticStep],
        weights: list[float] | None = None,
    ):
        self.steps = steps
        self.weights = weights or [1 for _ in steps]
        assert len(self.steps) == len(self.weights)

    def compute_ranges(self, population, target_size):
        """Computes the ranges for each slide, according to weights."""
        total = sum(self.weights)
        indices = [0] + self.cumsum(
            [int(round(w * len(population) / total, 0)) for w in self.weights],
        )
        ranges = list(zip(indices, indices[1:]))
        if ranges[-1][0] < target_size:
            ranges[-1] = (ranges[-1][0], target_size)
        return ranges

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        ranges = self.compute_ranges(population, target_size)
        assert len(ranges) == len(self.steps)

        retlist = self.concat(
            [
                step.iterate(
                    problem,
                    evaluator,
                    representation,
                    random_source,
                    deepcopy(population),
                    end - start,
                    generation,
                )
                for ((start, end), step) in zip(ranges, self.steps)
                if end - start > 0
            ],
        )
        assert len(retlist) == target_size
        return retlist

    def concat(self, ls):
        rl = []
        for li in ls:
            rl.extend(li)
        return rl

    def cumsum(self, li):
        v = 0
        nl = []
        for i in li:
            v = v + i
            nl.append(v)
        return nl

    def __str__(self):
        return "||".join([f"({x})" for x in self.steps])


class ExclusiveParallelStep(ParallelStep):
    """A version of ParallelStep, where each parallel step receives a portion
    of the input population equal to the target population of each slice."""

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
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
                    evaluator,
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
