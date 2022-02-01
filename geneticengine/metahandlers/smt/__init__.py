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
from geneticengine.metahandlers.smt.lang import dNode, dVar, dLit

import z3
import random

from geneticengine.core.representations.tree.treebased import is_metahandler


def simplify_type(t: Type) -> Type:
    if is_metahandler(t):
        return get_args(t)[0]
    return t


class SMT(MetaHandlerGenerator):
    def __init__(self, restriction_as_str="true==true"):
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

        ident = c["_"]
        treebased.SMTResolver.register_type(ident, base_type)

        treebased.SMTResolver.add_clause(
            [lambda types: self.restriction.translate(c, types)], {}
        )

        if base_type == int or base_type == bool or base_type == float:
            # we need the result, add receiver
            treebased.SMTResolver.add_clause(
                [],
                {ident: rec},
            )
        else:
            # just synth normally
            new_symbol(base_type, rec, depth, ident, context)

    def __repr__(self):
        return f"{self.restriction}"
