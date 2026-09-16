"""Regression tests for LexicaseSelection empty-candidate guards.

Rule 1: if the evaluator yields no (or too few) candidates, refill from the
original population with invalid fitness instead of crashing.
Rule 2: if filtering on a case would empty the pool (e.g. all-NaN objectives),
keep the pre-filter set.
"""

from __future__ import annotations

from geneticengine.algorithms.gp.operators.selection import LexicaseSelection
from geneticengine.problems import Fitness, MultiObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.solutions.individual import Individual


class _FakeIndividual(Individual[object]):
    def __init__(self, components: list[float], valid: bool = True):
        super().__init__()
        self._components = components
        self._valid = valid

    def get_phenotype(self) -> object:
        return self

    def seed_fitness(self, problem: MultiObjectiveProblem) -> None:
        self.set_fitness(problem, Fitness(self._components, valid=self._valid))


class _PassthroughEvaluator:
    def evaluate(self, problem, population):
        return list(population)


class _DropAllEvaluator:
    def evaluate(self, problem, population):
        return []


class _KeepFewEvaluator:
    def __init__(self, keep: int):
        self.keep = keep

    def evaluate(self, problem, population):
        return list(population)[: self.keep]


def _problem() -> MultiObjectiveProblem:
    return MultiObjectiveProblem(fitness_function=lambda _x: [0.0, 0.0], minimize=[True, True])


def test_rule1_refills_when_evaluator_drops_everyone():
    problem = _problem()
    inds = [_FakeIndividual([1.0, 2.0]) for _ in range(5)]
    # Individuals have no fitness yet; refill assigns invalid fitness.
    sel = LexicaseSelection(epsilon=False)
    winners = list(
        sel.iterate(
            problem,
            _DropAllEvaluator(),
            None,  # type: ignore[arg-type]
            NativeRandomSource(0),
            iter(inds),
            target_size=3,
            generation=0,
        )
    )
    assert len(winners) == 3
    assert all(not w.get_fitness(problem).valid for w in winners)


def test_rule1_refills_when_pool_depletes_before_target_size():
    problem = _problem()
    inds = [_FakeIndividual([float(i), float(i)]) for i in range(10)]
    for ind in inds:
        ind.seed_fitness(problem)
    sel = LexicaseSelection(epsilon=False)
    winners = list(
        sel.iterate(
            problem,
            _KeepFewEvaluator(keep=2),
            None,  # type: ignore[arg-type]
            NativeRandomSource(1),
            iter(inds),
            target_size=5,
            generation=0,
        )
    )
    assert len(winners) == 5


def test_rule2_nan_filter_does_not_empty_pool():
    problem = _problem()
    inds = [
        _FakeIndividual([float("nan"), 1.0]),
        _FakeIndividual([float("nan"), 2.0]),
        _FakeIndividual([float("nan"), 0.5]),
    ]
    for ind in inds:
        ind.seed_fitness(problem)
    sel = LexicaseSelection(epsilon=False)
    winners = list(
        sel.iterate(
            problem,
            _PassthroughEvaluator(),
            None,  # type: ignore[arg-type]
            NativeRandomSource(0),
            iter(inds),
            target_size=1,
            generation=0,
        )
    )
    assert len(winners) == 1
    assert winners[0] in inds


def test_normal_lexicase_still_prefers_better_case():
    problem = _problem()
    worse = _FakeIndividual([10.0, 10.0])
    better = _FakeIndividual([0.0, 0.0])
    for ind in (worse, better):
        ind.seed_fitness(problem)
    sel = LexicaseSelection(epsilon=False)
    winners = list(
        sel.iterate(
            problem,
            _PassthroughEvaluator(),
            None,  # type: ignore[arg-type]
            NativeRandomSource(0),
            iter([worse, better]),
            target_size=1,
            generation=0,
        )
    )
    assert winners == [better]
