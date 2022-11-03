from __future__ import annotations

from abc import ABCMeta
from pickle import _Pickler as StockPickler

from dill import register  # pyright: reportMissingImports=false
from pathos.multiprocessing import (
    ProcessingPool as Pool,
)  # pyright: reportMissingImports=false

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


@register(ABCMeta)
def save_abc(pickler, obj):
    StockPickler.save_type(pickler, obj)


class ParallelEvaluationStep(GeneticStep):
    """Pre-computes fitness in parallel."""

    def iterate(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
    ) -> list[Individual]:
        with Pool(len(population)) as pool:
            pool.map(lambda x: x.evaluate(problem), population)

        return population
