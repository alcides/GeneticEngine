import copy
from dataclasses import dataclass

import pytest
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import (
    DynamicStructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.grammatical_evolution.ge import GrammaticalEvolutionRepresentation
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    StructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.stackgggp import StackBasedGGGPRepresentation
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.core.representations.tree_smt.treebased import SMTTreeBasedRepresentation


@dataclass
class X:
    i: int
    j: int


def test_random_int():
    r = RandomSource(1)
    v1 = r.randint(0, 10)
    r = RandomSource(2)
    v2 = r.randint(0, 10)
    r = RandomSource(1)
    v3 = r.randint(0, 10)
    assert v1 == v3
    assert v1 != v2


def test_random_bool():
    r = RandomSource(1)
    v1 = [r.random_bool() for _ in range(100)]
    r = RandomSource(2)
    v2 = [r.random_bool() for _ in range(100)]
    r = RandomSource(1)
    v3 = [r.random_bool() for _ in range(100)]
    assert v1 == v3
    assert v1 != v2


def test_random_float():
    r = RandomSource(1)
    v1 = r.random_float(0, 10)
    r = RandomSource(2)
    v2 = r.random_float(0, 10)
    r = RandomSource(1)
    v3 = r.random_float(0, 10)
    assert v1 == v3
    assert v1 != v2


def test_random_norm():
    r = RandomSource(1)
    v1 = [r.normalvariate(3, 4) for _ in range(100)]
    r = RandomSource(2)
    v2 = [r.normalvariate(3, 4) for _ in range(100)]
    r = RandomSource(1)
    v3 = [r.normalvariate(3, 4) for _ in range(100)]
    assert v1 == v3
    assert v1 != v2


def test_random_choice():
    r = RandomSource(1)
    v1 = r.choice([n for n in range(0, 1000)])
    r = RandomSource(2)
    v2 = r.choice([n for n in range(0, 1000)])
    r = RandomSource(1)
    v3 = r.choice([n for n in range(0, 1000)])
    assert v1 == v3
    assert v1 != v2


def test_random_choice_weighted():
    r = RandomSource(1)
    v1 = r.choice_weighted([n for n in range(0, 1000)], [n for n in range(0, 1000)])
    r = RandomSource(2)
    v2 = r.choice_weighted([n for n in range(0, 1000)], [n for n in range(0, 1000)])
    r = RandomSource(1)
    v3 = r.choice_weighted([n for n in range(0, 1000)], [n for n in range(0, 1000)])
    assert v1 == v3
    assert v1 != v2


def test_random_shuffle():
    original = [n for n in range(100)]

    r = RandomSource(1)
    v1 = copy.copy(original)
    v1 = r.shuffle(v1)
    v1 = r.shuffle(v1)

    r = RandomSource(2)
    v2 = copy.copy(original)
    v2 = r.shuffle(v2)
    v2 = r.shuffle(v2)

    r = RandomSource(1)
    v3 = copy.copy(original)
    v3 = r.shuffle(v3)
    v3 = r.shuffle(v3)

    assert v1 == v3
    assert v1 != v2


def test_random_pop():
    original = [n for n in range(100)]

    r = RandomSource(1)
    l1 = copy.copy(original)
    v1 = r.pop_random(l1)
    v1 += r.pop_random(l1)

    r = RandomSource(2)
    l2 = copy.copy(original)
    v2 = r.pop_random(l2)
    v2 += r.pop_random(l2)

    r = RandomSource(1)
    l3 = copy.copy(original)
    v3 = r.pop_random(l3)
    v3 += r.pop_random(l3)

    assert v1 == v3
    assert v1 != v2


@pytest.mark.parametrize(
    "representation",
    [
        TreeBasedRepresentation,
        SMTTreeBasedRepresentation,
        GrammaticalEvolutionRepresentation,
        StructuredGrammaticalEvolutionRepresentation,
        DynamicStructuredGrammaticalEvolutionRepresentation,
        StackBasedGGGPRepresentation,
    ],
)
def test_random_gp(representation):
    grammar = extract_grammar([X], X)

    vals = []
    for _ in range(2):
        gp = GP(
            problem=SingleObjectiveProblem(lambda x: x.i * x.j),
            representation=representation(grammar, max_depth=3),
            random_source=RandomSource(0),
            stopping_criterium=GenerationStoppingCriterium(10),
        )
        e = gp.evolve()
        v = e.get_phenotype().i, e.get_phenotype().j
        vals.append(v)
    print(vals)
    assert vals[0] == vals[1]
