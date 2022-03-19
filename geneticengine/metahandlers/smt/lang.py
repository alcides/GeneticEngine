from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

import z3


class dNode:
    def translate(self, mappings: dict[str, str], types: dict[str, Any]):
        raise NotImplementedError

    def eval(self, x):
        raise NotImplementedError

    def collect_vars(self):
        raise NotImplementedError


class dLit(dNode):
    def __init__(self, val):
        self.val = val

    def translate(self, mappings, types):
        return self.val

    def eval(self, x):
        return self.val

    def __str__(self):
        return str(self.val)

    def collect_vars(self):
        return []


def s(x):
    """
    Sanitize ints
    :param x: to sanitize
    :return: sanitized
    """
    if isinstance(x, int) or isinstance(x, float):
        return dLit(x)

    return x


class dVar(dNode):

    # Name needed for multiple variables
    def __init__(self, names: list[str]):
        self.names = names

    def translate(self, mappings, types):
        resolved = mappings[self.names[0]]
        fullname = "_".join([resolved] + self.names[1:])
        cons = types[fullname]
        return cons(fullname)

    def eval(self, x):
        return x

    def __str__(self):
        return ".".join(self.names)

    def collect_vars(self):
        return [self.translate()]


class dAnd(dNode):
    def __init__(self, left, right):
        self.left = s(left)
        self.right = s(right)

    def translate(self, mappings, types):
        return z3.And(
            self.left.translate(
                mappings,
                types,
            ),
            self.right.translate(mappings, types),
        )

    def eval(self, x):
        return self.left.eval(x) and self.right.eval(x)

    def __str__(self):
        return f"{self.left} && {self.right}"

    def collect_vars(self):
        return self.left.collect_vars() + self.right.collect_vars()


class dOr(dNode):
    def __init__(self, left, right):
        self.left = s(left)
        self.right = s(right)

    def translate(self, mappings, types):
        return z3.Or(
            self.left.translate(
                mappings,
                types,
            ),
            self.right.translate(mappings, types),
        )

    def eval(self, x):
        return self.left.eval(x) or self.right.eval(x)

    def __str__(self):
        return f"{self.left} || {self.right}"

    def collect_vars(self):
        return self.left.collect_vars() + self.right.collect_vars()


class dLE(dNode):
    def __init__(self, left, right):
        self.left = s(left)
        self.right = s(right)

    def translate(self, mappings, types):
        return self.left.translate(mappings, types) <= self.right.translate(
            mappings,
            types,
        )

    def eval(self, x):
        return self.left.eval(x) <= self.right.eval(x)

    def __str__(self):
        return f"{self.left} <= {self.right}"

    def collect_vars(self):
        return self.left.collect_vars() + self.right.collect_vars()


class dLt(dNode):
    def __init__(self, left, right):
        self.left = s(left)
        self.right = s(right)

    def translate(self, mappings, types):
        return self.left.translate(mappings, types) < self.right.translate(
            mappings,
            types,
        )

    def eval(self, x):
        return self.left.eval(x) < self.right.eval(x)

    def __str__(self):
        return f"{self.left} < {self.right}"

    def collect_vars(self):
        return self.left.collect_vars() + self.right.collect_vars()


class dGE(dNode):
    def __init__(self, left, right):
        self.left = s(left)
        self.right = s(right)

    def translate(self, mappings, types):
        return self.left.translate(mappings, types) >= self.right.translate(
            mappings,
            types,
        )

    def eval(self, x):
        return self.left.eval(x) >= self.right.eval(x)

    def __str__(self):
        return f"{self.left} >= {self.right}"

    def collect_vars(self):
        return self.left.collect_vars() + self.right.collect_vars()


class dGt(dNode):
    def __init__(self, left, right):
        self.left = s(left)
        self.right = s(right)

    def translate(self, mappings, types):
        return self.left.translate(mappings, types) > self.right.translate(
            mappings,
            types,
        )

    def eval(self, x):
        return self.left.eval(x) > self.right.eval(x)

    def __str__(self):
        return f"{self.left} > {self.right}"

    def collect_vars(self):
        return self.left.collect_vars() + self.right.collect_vars()


class dEQ(dNode):
    def __init__(self, left, right):
        self.left = s(left)
        self.right = s(right)

    def translate(self, mappings, types):
        return self.left.translate(mappings, types) == self.right.translate(
            mappings,
            types,
        )

    def eval(self, x):
        return self.left.eval(x) == self.right.eval(x)

    def __str__(self):
        return f"{self.left} == {self.right}"

    def collect_vars(self):
        return self.left.collect_vars() + self.right.collect_vars()


class dMod(dNode):
    def __init__(self, left, right):
        self.left = s(left)
        self.right = s(right)

    def translate(self, mappings, types):
        return self.left.translate(mappings, types) % self.right.translate(
            mappings,
            types,
        )

    def eval(self, x):
        return self.left.eval(x) % self.right.eval(x)

    def __str__(self):
        return f"{self.left} % {self.right}"

    def collect_vars(self):
        return self.left.collect_vars() + self.right.collect_vars()


class dPlus(dNode):
    def __init__(self, left, right):
        self.left = s(left)
        self.right = s(right)

    def translate(self, mappings, types):
        return self.left.translate(mappings, types) + self.right.translate(
            mappings,
            types,
        )

    def eval(self, x):
        return self.left.eval(x) + self.right.eval(x)

    def __str__(self):
        return f"({self.left} + {self.right})"

    def collect_vars(self):
        return self.left.collect_vars() + self.right.collect_vars()


class dNEQ(dNode):
    def __init__(self, left, right):
        self.left = s(left)
        self.right = s(right)

    def translate(self, mappings, types):
        return self.left.translate(mappings, types) != self.right.translate(
            mappings,
            types,
        )

    def eval(self, x):
        return self.left.eval(x) != self.right.eval(x)

    def __str__(self):
        return f"{self.left} != {self.right}"

    def collect_vars(self):
        return self.left.collect_vars() + self.right.collect_vars()


class dNot(dNode):
    def __init__(self, cond):
        self.cond = s(cond)

    def translate(self, mappings, types):
        return z3.Not(self.cond.translate(mappings, types))

    def eval(self, x):
        return not self.cond.eval(x)

    def __str__(self):
        return f"!({self.cond})"

    def collect_vars(self):
        return self.cond.collect_vars()


class fNavigate(dNode):
    def __init__(self, obj, expr):
        self.obj = obj
        self.expr = expr

    def translate_mappings(self, mappings: dict[str, str]):
        ret = {}
        obj = self.obj
        for k, v in mappings.items():
            if k.startswith(obj + "."):
                newk = k[len(obj) + 1 :]
            else:
                newk = "__parent__." + k
            ret[newk] = v
        return ret

    def translate(self, mappings: dict[str, str], types):
        mappings = self.translate_mappings(mappings)

        return self.expr.translate(mappings, types)

    def eval(self, x):
        raise NotImplementedError()

    def __str__(self):
        return self.obj + "." + str(self.expr)

    def collect_vars(self):
        raise NotImplementedError()


class dAllPairs(dNode):
    def __init__(self, cont: str, x: str, y: str, block: dNode):
        self.cont = cont
        self.x = x
        self.y = y
        self.block = block

    def _gen_pairs(self, l: list):
        for i1, e1 in enumerate(l):
            for i2, e2 in enumerate(l):
                if i1 != i2:
                    yield e1, e2

    def translate(self, mappings: dict[str, str], types: dict[str, Any]):
        real_cont = mappings[self.cont]
        subs = []
        for k in types.keys():
            if not k.startswith(real_cont):
                continue
            trim_k = k[len(real_cont) :]
            if "." in trim_k:
                continue
            subs.append(k)

        ret = True

        for x, y in self._gen_pairs(subs):
            maps = mappings.copy()
            maps[self.x] = x
            maps[self.y] = y
            transed = self.block.translate(maps, types)
            ret = z3.And(ret, transed)

        return ret
