import copy
from dataclasses import dataclass

import pytest
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.grammatical_evolution.dynamic_structured_ge import (
    DynamicStructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.representations.grammatical_evolution.ge import GrammaticalEvolutionRepresentation
from geneticengine.representations.grammatical_evolution.structured_ge import (
    StructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.representations.stackgggp import StackBasedGGGPRepresentation
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


@dataclass
class X:
    i: int
    j: int


def test_random_int():
    r = NativeRandomSource(1)
    v1 = r.randint(0, 10)
    r = NativeRandomSource(2)
    v2 = r.randint(0, 10)
    r = NativeRandomSource(1)
    v3 = r.randint(0, 10)
    assert v1 == v3
    assert v1 != v2


def test_random_bool():
    r = NativeRandomSource(1)
    v1 = [r.random_bool() for _ in range(100)]
    r = NativeRandomSource(2)
    v2 = [r.random_bool() for _ in range(100)]
    r = NativeRandomSource(1)
    v3 = [r.random_bool() for _ in range(100)]
    assert v1 == v3
    assert v1 != v2


def test_random_float():
    r = NativeRandomSource(1)
    v1 = r.random_float(0, 10)
    r = NativeRandomSource(2)
    v2 = r.random_float(0, 10)
    r = NativeRandomSource(1)
    v3 = r.random_float(0, 10)
    assert v1 == v3
    assert v1 != v2


def test_random_norm():
    r = NativeRandomSource(1)
    v1 = [r.normalvariate(3, 4) for _ in range(100)]
    r = NativeRandomSource(2)
    v2 = [r.normalvariate(3, 4) for _ in range(100)]
    r = NativeRandomSource(1)
    v3 = [r.normalvariate(3, 4) for _ in range(100)]
    assert v1 == v3
    assert v1 != v2


def test_random_choice():
    r = NativeRandomSource(1)
    v1 = r.choice([n for n in range(0, 1000)])
    r = NativeRandomSource(2)
    v2 = r.choice([n for n in range(0, 1000)])
    r = NativeRandomSource(1)
    v3 = r.choice([n for n in range(0, 1000)])
    assert v1 == v3
    assert v1 != v2


def test_random_choice_weighted():
    r = NativeRandomSource(1)
    v1 = r.choice_weighted([n for n in range(0, 1000)], [n for n in range(0, 1000)])
    r = NativeRandomSource(2)
    v2 = r.choice_weighted([n for n in range(0, 1000)], [n for n in range(0, 1000)])
    r = NativeRandomSource(1)
    v3 = r.choice_weighted([n for n in range(0, 1000)], [n for n in range(0, 1000)])
    assert v1 == v3
    assert v1 != v2


def test_random_shuffle():
    original = [n for n in range(100)]

    r = NativeRandomSource(1)
    v1 = copy.copy(original)
    v1 = r.shuffle(v1)
    v1 = r.shuffle(v1)

    r = NativeRandomSource(2)
    v2 = copy.copy(original)
    v2 = r.shuffle(v2)
    v2 = r.shuffle(v2)

    r = NativeRandomSource(1)
    v3 = copy.copy(original)
    v3 = r.shuffle(v3)
    v3 = r.shuffle(v3)

    assert v1 == v3
    assert v1 != v2


def test_random_pop():
    original = [n for n in range(100)]

    r = NativeRandomSource(1)
    l1 = copy.copy(original)
    v1 = r.pop_random(l1)
    v1 += r.pop_random(l1)

    r = NativeRandomSource(2)
    l2 = copy.copy(original)
    v2 = r.pop_random(l2)
    v2 += r.pop_random(l2)

    r = NativeRandomSource(1)
    l3 = copy.copy(original)
    v3 = r.pop_random(l3)
    v3 += r.pop_random(l3)

    assert v1 == v3
    assert v1 != v2


@pytest.mark.parametrize(
    "representation",
    [
        lambda g, r, depth: TreeBasedRepresentation(g, decider=MaxDepthDecider(r, g, max_depth=depth)),
        lambda g, r, depth: GrammaticalEvolutionRepresentation(g, decider=MaxDepthDecider(r, g, max_depth=depth)),
        lambda g, r, depth: StructuredGrammaticalEvolutionRepresentation(
            g,
            decider=MaxDepthDecider(r, g, max_depth=depth),
        ),
        lambda g, r, depth: DynamicStructuredGrammaticalEvolutionRepresentation(g, max_depth=depth),
        lambda g, r, depth: StackBasedGGGPRepresentation(g),
    ],
)
def test_random_gp(representation):
    grammar = extract_grammar([X], X)

    vals = []
    for _ in range(2):
        r = NativeRandomSource(0)
        repr = representation(grammar, r, depth=3)
        gp = GeneticProgramming(
            problem=SingleObjectiveProblem(lambda x: x.i * x.j),
            budget=EvaluationBudget(100),
            population_size=10,
            representation=repr,
            random=r,
        )
        es = gp.search()
        e = es[0]
        v = e.get_phenotype().i, e.get_phenotype().j
        vals.append(v)
    assert vals[0] == vals[1]
