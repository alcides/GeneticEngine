from __future__ import annotations

from typing import Any, get_args


from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.utils import is_metahandler
from geneticengine.random.sources import RandomSource
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator
from geneticengine.grammar.metahandlers.smt.parser import p_expr


def simplify_type(t: type) -> type:
    if is_metahandler(t):
        return get_args(t)[0]
    return t


class SMT(MetaHandlerGenerator):
    def __init__(self, restriction_as_str="true==true"):
        self.restriction_as_str = restriction_as_str
        self.restriction = p_expr(restriction_as_str)

    def validate(self, v) -> bool:
        return True  # TODO: SMT

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Any,
        dependent_values: dict[str, Any],
        parent_values: list[dict[str, Any]],
    ):
        return base_type
        # # fix_types(self.restriction, context)
        # c = context.copy()

        # ident = c["_"]
        # # smt.SMTResolver.register_type(ident, base_type)

        # # smt.SMTResolver.add_clause(
        # #     [lambda types: self.restriction.translate(c, types)],
        # #     {},
        # # )

        # if base_type == int or base_type == bool or base_type == float:
        #     # we need the result, add receiver
        #     # smt.SMTResolver.add_clause(
        #     #     [],
        #     #     {ident: rec},
        #     # )
        #     pass
        # else:
        #     # just synth normally
        #     new_symbol(base_type, rec, depth, ident, context)

    def __repr__(self):
        return f"{self.restriction}"
