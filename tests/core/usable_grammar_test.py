from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypeVar

from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator
from geneticengine.random.sources import RandomSource

T = TypeVar("T")


class NoOp(MetaHandlerGenerator):
    def validate(self, Any) -> bool:
        return True

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
    ) -> Any:
        return rec(base_type)


class Root(ABC):
    pass


@dataclass
class A(Root):
    x: int


@dataclass
class IJ:
    x: int


@dataclass
class H:
    x: int


@dataclass
class G(Root):
    x: tuple[H, IJ]


@dataclass
class F:
    x: int


@dataclass
class E(Root):
    fs: list[F]


@dataclass
class D:
    x: int


@dataclass
class C:
    d: Annotated[D, NoOp]


@dataclass
class B(Root):
    c: C


@dataclass
class Z:
    pass


@dataclass
class X:
    k: int
    z: Z


class Y(ABC):
    pass


@dataclass
class W(Root):
    y: Y


def test_useful_grammar():
    positives = [A, B, C, D, E, F, G, H, IJ]
    negatives = [Z, X, Y, W]
    g = extract_grammar(positives + negatives, Root)
    g2 = g.usable_grammar()

    symbols = g2.considered_subtypes
    print(dir(g2))
    for a in positives:
        assert a in symbols
    for a in negatives:
        assert a not in symbols
