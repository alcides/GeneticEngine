from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.stop import EvaluationLimitCriterium

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass(unsafe_hash=True)
class Option(Root):
    a: int


def fitness_function(r: Root) -> float:
    assert isinstance(r, Option)
    return r.a


class TestStoppingCriteria:
    def test_evaluations(self):

        limit = 120
        population_size = 15

        grammar = extract_grammar([Option], Root)
        gp = GP(
            representation=TreeBasedRepresentation(grammar=grammar, max_depth=2),
            stopping_criterium=EvaluationLimitCriterium(limit),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            population_size=population_size,
        )
        gp.evolve()

        assert gp.evaluator.get_count() < limit + 2 * population_size
