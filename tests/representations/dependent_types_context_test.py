from abc import ABC
from dataclasses import dataclass
import string
from typing import Annotated, Any, Callable, TypeVar


from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator
from geneticengine.grammar.metahandlers.dependent import Dependent
from geneticengine.grammar.metahandlers.vars import VarRange
from geneticengine.random.sources import NativeRandomSource, RandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.solutions.tree import GengyList


T = TypeVar("T")


class IdMH(MetaHandlerGenerator):
    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
        parent_values: list[dict[str,Any]],
    ):
        return rec(base_type)

    def validate(self, v) -> bool:
        return True


class AnyContext(MetaHandlerGenerator):
    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
        parent_values: list[dict[str,Any]],
    ):
        # Always creates an empty list.
        # Genetic Engine requires a specific wrapper (GengyList) with meta-information
        return GengyList(str, [])

    def validate(self, v) -> bool:
        return True


class ContextMH(MetaHandlerGenerator):
    """This methandler takes a context, and injects it in the generated
    value."""

    def __init__(self, ctx: list[str]):
        self.ctx = ctx

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec,  # : Callable[[type[T]], T] TODO: add kwargs
        dependent_values: dict[str, Any],
        parent_values: list[dict[str,Any]],
    ):
        v = rec(base_type, initial_values={"ctx": self.ctx})
        return v

    def validate(self, v) -> bool:
        return True  # TODO: all free variables are in context


class Expr(ABC):
    pass


@dataclass
class Literal(Expr):
    v: Annotated[int, IntRange(0, 3)]


@dataclass
class Let(Expr):
    ctx: Annotated[list[str], AnyContext()]
    name: Annotated[str, VarRange(string.ascii_lowercase)]
    body: Annotated[Expr, Dependent("ctx,name", lambda ctx, name: ContextMH(ctx + [name]))]


@dataclass
class Var(Expr):
    ctx: Annotated[list[str], AnyContext()]
    name: Annotated[str, Dependent("ctx", lambda ctx: VarRange(ctx))]


def typecheck(ctx: list[str], e: Expr) -> bool:
    match e:
        case Literal(v=_):
            return True
        case Let(ctx=_, name=n, body=b):
            return typecheck(ctx + [n], b)
        case Var(ctx=_, name=n):
            return n in ctx
    return True


def test_dependent_types_context():
    r = NativeRandomSource(seed=1)
    g = extract_grammar([Let, Var, Literal], Expr)
    decider = MaxDepthDecider(r, g, 5)

    repr = TreeBasedRepresentation(g, decider)

    for _ in range(10):
        el = repr.create_genotype(r)
        print(el)
        assert typecheck([], el)
        for _ in range(10):
            p = repr.mutate(r, el)
            assert typecheck([], p)
