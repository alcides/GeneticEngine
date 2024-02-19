# from __future__ import annotations

# from abc import ABC
# from dataclasses import dataclass
# from typing import Annotated
# from typing import TypeVar


# from geneticengine.grammar.grammar import extract_grammar
# from geneticengine.random.sources import NativeRandomSource
# from geneticengine.grammar.metahandlers.smt import SMT
# from geneticengine.representations.tree.treebased import random_node


# class Root(ABC):
#     pass


# @dataclass
# class IntC(Root):
#     x: Annotated[int, SMT("9 <= _ && _ <= 10")]


# @dataclass
# class BoolC(Root):
#     x: Annotated[bool, SMT("_")]


# @dataclass
# class FloatC(Root):
#     x: Annotated[float, SMT("_ > 0.3")]


# @dataclass
# class ComplexInts(Root):
#     a: int
#     b: Annotated[int, SMT("a == b + 1")]
#     c: Annotated[int, SMT("b == _ + 1")]


# @dataclass
# class ObjectNavigationChild:
#     val: int


# @dataclass
# class ObjectNavigation(Root):
#     val: Annotated[int, SMT("_ == child.val + 2020")]
#     child: ObjectNavigationChild


# @dataclass
# class Comprehension(Root):
#     vals: Annotated[
#         list[Annotated[int, SMT()]],
#         SMT(
#             "AllPairs(_, x, y){x != y}",
#         ),
#     ]


# T = TypeVar("T")


# class TestMetaHandler:
#     def skeleton(self, *t, depth=3):
#         r = NativeRandomSource(seed=1)
#         g = extract_grammar(list(t), Root)
#         n = random_node(r, g, depth, Root)
#         assert isinstance(n, Root)
#         return n

#     def test_int(self):
#         n = self.skeleton(IntC)
#         assert isinstance(n, IntC)
#         assert 9 <= n.x <= 10

#     def test_complex_int(self):
#         n = self.skeleton(ComplexInts)
#         assert isinstance(n, ComplexInts)
#         assert n.a == n.b + 1
#         assert n.b == n.c + 1

#     def test_bool(self):
#         n = self.skeleton(BoolC)
#         assert isinstance(n, BoolC)
#         assert n.x is True

#     def test_real(self):
#         n = self.skeleton(FloatC)
#         assert isinstance(n, FloatC)
#         assert n.x > 0.3

#     def test_object_navigation(self):
#         n = self.skeleton(ObjectNavigation, ObjectNavigationChild, depth=4)
#         assert isinstance(n, ObjectNavigation)
#         assert n.val == n.child.val + 2020

#     def test_comprehensions(self):
#         n = self.skeleton(Comprehension, depth=10)
#         assert isinstance(n, Comprehension)
#         seen = set()
#         for num in n.vals:
#             assert num not in seen
#             seen.add(num)
