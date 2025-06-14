from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypeVar

from geneticengine.grammar.grammar import Grammar, all_with_recursion, extract_grammar
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
        parent_values: list[dict[str, Any]],
    ) -> Any:
        return rec(base_type)


class Root(ABC):
    pass


@dataclass
class A(Root):
    x: int


@dataclass
class K(ABC):
    pass


@dataclass
class N(K):
    n: K


class Loop(ABC):
    pass


@dataclass
class LoopGo(Loop):
    loop: Loop


@dataclass
class Ok(K):
    n: Loop


@dataclass
class L(K):
    v: int


@dataclass
class M(Root):
    k: K


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
    r: Root


class Y(ABC):
    pass


@dataclass
class W(Root):
    y: Y


positives = [A, B, C, D, E, F, G, H, IJ, K, L, M, N]
negatives = [Z, X, Y, W, Ok, Loop, LoopGo]


def test_useful_grammar():

    g = extract_grammar(positives + negatives, Root)
    g2 = g.usable_grammar()

    symbols = g2.considered_subtypes
    for a in positives:
        assert a in symbols
    for a in negatives:
        assert a not in symbols


def test_reaches():
    g = extract_grammar(positives + negatives, Root)

    for p in positives:
        assert g.is_reachable(Root, p)
        assert g.reaches_leaf(p)

    for n in negatives:
        assert not g.is_reachable(Root, n) or not g.reaches_leaf(n)


def test_all_with_recursion():
    assert all_with_recursion([True])
    assert all_with_recursion([True, True])
    assert all_with_recursion([True, True, None])
    assert all_with_recursion([True, None, True, None])
    assert not all_with_recursion([False])
    assert not all_with_recursion([False, True])
    assert not all_with_recursion([None])
    assert not all_with_recursion([None, False])
