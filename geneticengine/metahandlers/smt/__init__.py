from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Protocol,
    Type,
    TypeVar,
    ForwardRef,
    Tuple,
    get_args,
)

from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator

from geneticengine.core.grammar import Grammar
from geneticengine.metahandlers.smt.parser import p_expr
from geneticengine.metahandlers.smt.lang import dNode, dVar

import z3
import random

from geneticengine.core.representations.treebased import is_metahandler


def simplify_type(t: Type) -> Type:
    if is_metahandler(t):
        return get_args(t)[0]
    return t


z3types = {int: "int", bool: "bool", float: "real"}


def fix_types(e: dNode, context: Dict[str, Type]):
    if isinstance(e, dVar):
        e.type = z3types[simplify_type(context[e.name])]
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
        wrapper: Callable[[Any, str, int, Callable[[int], Any]], Any],
        rec: Any,
        depth: int,
        base_type,
        argname: str,
        context: Dict[str, Type],
    ):
        fix_types(self.restriction, context)

        seed = r.randint(0, 1000000)
        n_samples = 10000

        solver = z3.Solver()
        random.seed(seed)
        solver.set(":random-seed", random.randint(0, n_samples * 100))
        solver.reset()

        keys = self.restriction.collect_vars()
        k = [k for k in keys if str(k) == argname][0]
        restr = self.restriction.translate()

        solver.add(restr)
        res = solver.check()

        if res != z3.sat:
            raise Exception(f"{restr} failed with {seed} {res}")

        model = solver.model()
        evaled = model.eval(k, model_completion=True)

        if type(evaled) == z3.z3.BoolRef:
            evaled = bool(str(evaled))
        elif type(evaled) == z3.z3.IntNumRef:
            evaled = int(str(evaled))
        else:
            evaled = eval(str(evaled))
        return evaled

    def __repr__(self):
        return f"{self.restriction}"
