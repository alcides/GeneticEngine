import numpy as np
import pytest
from typing import List
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import MultiObjectiveProblem, Fitness
from geneticengine.solutions.individual import PhenotypicIndividual

# Delayed import to avoid circular import issues
import importlib

class DummyProblem(MultiObjectiveProblem):
    def __init__(self, minimize: list[bool]):
        super().__init__(fitness_function=None, minimize=minimize)

    def number_of_objectives(self) -> int:
        return 3
    def is_better(self, f1: Fitness, f2: Fitness) -> bool:
        return f1.fitness_components > f2.fitness_components


class DummyIndividual(PhenotypicIndividual):
    def __init__(self, scores: List[float]):
        self._scores = scores

    def get_fitness(self, problem):
        class DummyFitness:
            def __init__(self, scores):
                self.fitness_components = scores

            def __repr__(self):
                return str(self.fitness_components)

        return DummyFitness(self._scores)

    def has_fitness(self, problem) -> bool:
        return True


@pytest.mark.parametrize(
    "max_sample_size, percent, target_size", [
        (3, 1.0, 1),
        (2, 0.5, 2),
    ],
)
def test_informed_downsampling(max_sample_size, percent, target_size):
    # Lazy import avoids circular dependency issues
    selection_mod = importlib.import_module("geneticengine.algorithms.gp.operators.selection")
    InformedDownsamplingSelection = getattr(selection_mod, "InformedDownsamplingSelection")

    problem = DummyProblem(minimize=[False, False, False])
    evaluator = SequentialEvaluator()

    individuals = [
        DummyIndividual([0.9, 0.4, 0.2]),
        DummyIndividual([1.0, 0.9, 0.6]),
        DummyIndividual([0.7, 0.8, 1.0]),
        DummyIndividual([0.5, 0.1, 0.9]),
        DummyIndividual([0.3, 0.2, 0.5]),
    ]

    selector = InformedDownsamplingSelection(max_sample_size=max_sample_size, percent=percent)

    selected = list(
        selector.iterate(
            problem=problem,
            evaluator=evaluator,
            representation=None,
            random=np.random.default_rng(seed=123),
            population=iter(individuals),
            target_size=target_size,
            generation=0,
        ),
    )

    assert len(selected) == target_size
    assert all(isinstance(ind, DummyIndividual) for ind in selected)
