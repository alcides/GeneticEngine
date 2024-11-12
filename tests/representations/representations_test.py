from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Union

import pytest
from geneticengine.algorithms.enumerative import EnumerativeSearch, combine_list_types
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget, TimeBudget

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
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
from geneticengine.grammar.metahandlers.floats import FloatRange
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.grammar.metahandlers.vars import VarRange


class Root(ABC):
    pass


@dataclass
class Basic(Root):
    a: int
    b: float
    c: str


@dataclass
class UnionT(Root):
    x: Union[int, Concrete]


@dataclass
class IntRangeM(Root):
    x: Annotated[int, IntRange[9, 10]]


@dataclass
class Branch(Root):
    a: Root
    b: Root


@dataclass
class Concrete(Root):
    a: int


@dataclass
class FloatRangeM(Root):
    x: Annotated[float, FloatRange[9.0, 10.0]]


@dataclass
class VarRangeM(Root):
    x: Annotated[str, VarRange(["x", "y", "z"])]


@dataclass
class ListRangeM(Root):
    x: Annotated[list[int], ListSizeBetween[3, 4]]


@dataclass
class ListWrapper(Root):
    a: list[VarRangeM]


class TestRepresentation:
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
            lambda g, r, depth: StackBasedGGGPRepresentation(g, 100000, 10000),
        ],
    )
    def test_rep(self, representation) -> None:
        r = NativeRandomSource(seed=1)
        g: Grammar = extract_grammar([IntRangeM, ListRangeM, FloatRangeM, Branch, Concrete, ListWrapper], Root)

        def fitness_function(x: Root) -> float:
            return 0.5

        gp = GeneticProgramming(
            representation=representation(g, r, 10),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            random=r,
            budget=EvaluationBudget(1000),
        )
        gp.search()

    def test_enumerative(self) -> None:
        g: Grammar = extract_grammar([IntRangeM, ListRangeM, FloatRangeM, Branch, Concrete, ListWrapper], Root)

        def fitness_function(x: Root) -> float:
            return 0.5

        gp = EnumerativeSearch(
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            budget=TimeBudget(5),
            grammar=g,
        )
        gp.search()

    def test_enumerative_combine(self) -> None:
        def gen_bool(t, dependent_values):
            assert t is bool
            yield True
            yield False

        for m in combine_list_types([bool, bool], [], gen_bool):
            assert len(m) == 2
            for el in m:
                assert el in [True, False]
