from __future__ import annotations

from typing import Any
from typing import Callable

import z3


class SMTException(Exception):
    pass


# TODO: make non static
class SMTResolver:
    clauses: list[Any] = []
    receivers: dict[str, Callable] = {}
    types: dict[str, Callable] = {}

    @staticmethod
    def clean():
        SMTResolver.clauses = []
        SMTResolver.receivers = {}
        SMTResolver.types = {}

    @staticmethod
    def add_clause(claus, recs: dict[str, Callable]):
        SMTResolver.clauses.extend(claus)
        for k, v in recs.items():
            SMTResolver.receivers[k] = v

    @staticmethod
    def register_type(name, typ):
        try:
            SMTResolver.types[name] = SMTResolver.to_z3_typ(typ)
        except SMTException:
            pass

    @staticmethod
    def to_z3_typ(typ):
        if typ == bool:
            return z3.Bool
        elif typ == int:
            return z3.Int
        elif typ == float:
            return z3.Real
        raise SMTException("Could not encode in Z3")

    @staticmethod
    def resolve_clauses():

        if not SMTResolver.receivers:
            return  # don't try to smt solve if we don't need to

        solver = z3.Solver()

        solver.set(":random-seed", 1)
        solver.reset()

        for clause in SMTResolver.clauses:
            solver.add(clause(SMTResolver.types))
        res = solver.check()

        if res != z3.sat:
            raise Exception(f"{solver} failed with {res}")

        model = solver.model()
        for (name, recv) in SMTResolver.receivers.items():
            evaled = model.eval(
                SMTResolver.types[name](
                    name,
                ),
                model_completion=True,
            )

            recv(SMTResolver.get_type(evaled))

        SMTResolver.clauses = []
        SMTResolver.receivers = {}
        SMTResolver.types = {}

    @staticmethod
    def get_type(evaled):
        if type(evaled) == z3.z3.BoolRef:
            evaled = bool(str(evaled))
        elif type(evaled) == z3.z3.IntNumRef:
            evaled = int(str(evaled))
        elif type(evaled) == z3.z3.RatNumRef:
            evaled = eval(str(evaled))
        else:
            raise NotImplementedError(
                f"Don't know what to do with {type(evaled)} {evaled}",
            )
        return evaled

    @staticmethod
    def register_const(ident, val):
        SMTResolver.register_type(ident, type(val))
        ty = SMTResolver.types[ident]
        SMTResolver.clauses.append(lambda _: ty(ident) == val)
