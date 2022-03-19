from __future__ import annotations

from lark import Lark
from lark import Transformer
from lark import v_args

from geneticengine.metahandlers.smt.lang import *

dsl_grammar = """

expression : "-" expression_plus                             -> minus
           | expression_un                                   -> same

expression_un : expression_bool                              -> same
           | "!" expression_un                               -> nnot
           | expression_bool "==" expression_un              -> binop_eq
           | expression_bool "!=" expression_un              -> binop_neq
           | expression_bool "&&" expression_un              -> binop_and
           | expression_bool _DOUBLEPIPE expression_un       -> binop_or

expression_bool : expression_plus                            -> same
                | expression_plus "<" expression_bool        -> binop_lt
                | expression_plus "<=" expression_bool       -> binop_le
                | expression_plus ">" expression_bool        -> binop_gt
                | expression_plus ">=" expression_bool       -> binop_ge
                | expression_plus "-->" expression_bool      -> binop_imp

expression_plus : expression_fact                           -> same
                | expression_fact "+" expression_plus       -> binop_plus
                | expression_fact "-" expression_plus       -> binop_minus

expression_fact : expression_simple                         -> same
                | expression_simple "*" expression_fact     -> binop_mult
                | expression_simple "/" expression_fact     -> binop_div
                | expression_simple "%" expression_fact     -> binop_mod


expression_simple : "(" expression ")"                      -> same
            | comprehension                                 -> same
            | INTLIT                                        -> int_lit
            | SIGNED_INT                                    -> int_lit
            | FLOATLIT                                      -> float_lit
            | BOOLLIT                                       -> bool_lit
            | VAR ("." VAR)*                                -> var

comprehension : "AllPairs" "(" VAR "," VAR "," VAR ")" "{" expression "}" -> all_pairs

BOOLLIT.5 : "true" | "false"
INTLIT : /[0-9][0-9]*/
FLOATLIT : SIGNED_FLOAT
_DOUBLEPIPE.11 : "||"
VAR : (("a".."z")|"_"|("A".."Z")) (("0".."9")|("a".."z")|("A".."Z"))*

%import common.WS
%import common.CNAME
%import common.SIGNED_INT
%import common.SIGNED_FLOAT

%ignore WS
"""


class TreeToDSL(Transformer):
    def __init__(self):
        pass

    def same(self, args):
        return args[0]

    # Literals

    def var(self, args: list[str]):
        return dVar(args)

    def int_lit(self, args):
        return int(args[0])

    def float_lit(self, args):
        return float(args[0])

    def bool_lit(self, args):
        return str(args[0]) == "true"

    # Expressions

    def minus(self, args):
        raise NotImplementedError()

    def nnot(self, args):
        return dNot(*args)

    def binop_eq(self, args):
        return dEQ(*args)

    def binop_neq(self, args):
        return dNEQ(*args)

    def binop_and(self, args):
        return dAnd(*args)

    def binop_or(self, args):
        return dOr(*args)

    def binop_lt(self, args):
        return dLt(*args)

    def binop_le(self, args):
        return dLE(*args)

    def binop_gt(self, args):
        return dGt(*args)

    def binop_ge(self, args):
        return dGE(*args)

    def binop_imp(self, args):
        return dOr(dNot(args[0]), args[1])

    def binop_plus(self, args):
        return dPlus(*args)

    def binop_minus(self, args):
        raise NotImplementedError()

    def binop_mult(self, args):
        raise NotImplementedError()

    def binop_div(self, args):
        raise NotImplementedError()

    def binop_mod(self, args):
        return dMod(*args)

    def all_pairs(self, args):
        return dAllPairs(args[0], args[1], args[2], args[3])


def mk_parser():
    return Lark(
        dsl_grammar,
        parser="lalr",
        start="expression",
        transformer=TreeToDSL(),
    )


p_expr = mk_parser().parse
