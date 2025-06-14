from __future__ import annotations
from itertools import cycle
from typing import Any, Iterator, TypeVar

from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems.helpers import non_dominated
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.evaluation import Evaluator
from geneticengine.solutions.individual import PhenotypicIndividual

T = TypeVar("T")

def wrap(els:Iterator[T], size:int) -> Iterator[T]:
    for i, el in enumerate(cycle(els)):
        if i >= size:
            break
        yield el

class ElitismStep(GeneticStep):
    """Selects the best individuals from the population."""

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        candidates = evaluator.evaluate(problem, population)
        best : Iterator[PhenotypicIndividual[Any, Any]] = non_dominated(iter(candidates), problem)
        yield from wrap(iter(best), target_size)
