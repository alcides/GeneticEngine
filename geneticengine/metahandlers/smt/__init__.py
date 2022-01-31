from typing import (
    Any,
    Callable,
    Dict,
    Type,
    get_args,
)

from geneticengine.core.random.sources import Source
from geneticengine.core.representations.tree import treebased
from geneticengine.metahandlers.base import MetaHandlerGenerator

from geneticengine.core.grammar import Grammar
from geneticengine.metahandlers.smt.parser import p_expr
from geneticengine.metahandlers.smt.lang import dNode, dVar

import z3
import random

from geneticengine.core.representations.tree.treebased import is_metahandler


def simplify_type(t: Type) -> Type:
    if is_metahandler(t):
        return get_args(t)[0]
    return t


z3types = {int: "int", bool: "bool", float: "real"}


def fix_types(e: dNode, context: Dict[str, Type]):
    if isinstance(e, dVar):
        name = str(e)
        ty = simplify_type(context[name])
        e.type = z3types[ty]
    else:
        for p in dir(e):
            a = getattr(e, p)
            if isinstance(a, dNode):
                fix_types(a, context)


class SMT(MetaHandlerGenerator):
    def __init__(self, restriction_as_str):
        self.restriction_as_str = restriction_as_str
        self.restriction = p_expr(restriction_as_str)

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        context: Dict[str, str],
    ):
        # fix_types(self.restriction, context)
        c = context.copy()
        treebased.SMTResolver.add_clause(
            [lambda types: self.restriction.translate(c, types)],
            {c["_"]: rec},
        )

        treebased.SMTResolver.register_type(c["_"], base_type)

    def __repr__(self):
        return f"{self.restriction}"
