from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
import string
from typing import Annotated

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.grammar.metahandlers.parent import Parent
from geneticengine.grammar.metahandlers.vars import VarRange
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation

@dataclass
class Name:
    n: Annotated[str, VarRange(string.ascii_lowercase)]

class Expr(ABC):
    pass

@dataclass
class Sum(Expr):
    name: Annotated[Name, Parent("arguments", lambda a:VarRange(a))]
    pos1: Expr
    pos2: Expr

@dataclass
class Literal(Expr):
    v: Annotated[int, IntRange(0, 3)]

@dataclass
class Let(Expr):
    name: Annotated[Name, Parent("arguments", lambda a:VarRange(a))]
    val: Literal

@dataclass
class Lambda:
    name: Name
    arguments: Annotated[list[Name], ListSizeBetween(5,5)]
    exps: Annotated[list[Expr], ListSizeBetween(10,10)]

@dataclass
class Value:
    v: Annotated[int, Parent("values",lambda values : IntRange(values[-1].v+1,values[-1].v+1000) if values else IntRange(50,1000))]

@dataclass
class OrderList:
    values: Annotated[list[Value], ListSizeBetween(10,10)]

    def isOrdered(self):
        return all([v1.v < v2.v for v1, v2 in zip(self.values[:-1],self.values[1:])])

def typecheck(ctx: list[Name], e:Lambda|Expr) -> bool:
    match e:
        case Literal(v=_):
            return True
        case Let(name=n, val=val):
            return typecheck(ctx, val) and n in ctx
        case Sum(name=n, pos1=p1, pos2=p2):
            return n in ctx and typecheck(ctx, p1) and typecheck(ctx, p2)
        case Lambda(name=_, arguments=arguments, exps=exps):
            return False not in [typecheck(arguments,exp) for exp in exps]
    return True

def test_dependent_types_parent():
        r = NativeRandomSource(seed=1)
        g = extract_grammar([Lambda, Name, Sum, Let, Literal,Expr], Lambda)
        decider = MaxDepthDecider(r, g, 4)
        repr = TreeBasedRepresentation(g, decider=decider)

        for _ in range(10):
            el = repr.create_genotype(r)
            print(el)
            assert typecheck([], el)
            for _ in range(10):
                p = repr.mutate(r, el)
                assert typecheck([], p)

def test_dependent_types_parent_in_list():
        r = NativeRandomSource(seed=1)
        g = extract_grammar([Value, OrderList], OrderList)
        decider = MaxDepthDecider(r, g, 4)
        repr = TreeBasedRepresentation(g, decider=decider)

        for _ in range(10):
            el = repr.create_genotype(r)
            print(el)
            assert isinstance(el, OrderList)
            assert el.isOrdered()
            assert typecheck([], el)
