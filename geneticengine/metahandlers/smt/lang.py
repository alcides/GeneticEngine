import z3


class dNode:
    def translate(self, mappings, types):
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
    def __init__(self, name):
        self.name = name

    def translate(self, mappings, types):
        resolved = mappings[self.name]
        cons = types[resolved]
        return cons(resolved)

    def eval(self, x):
        return x

    def __str__(self):
        return self.name

    def collect_vars(self):
        return [self.translate()]


class dAnd(dNode):
    def __init__(self, left, right):
        self.left = s(left)
        self.right = s(right)

    def translate(self, mappings, types):
        return z3.And(
            self.left.translate(mappings, types), self.right.translate(mappings, types)
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
            self.left.translate(mappings, types), self.right.translate(mappings, types)
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
            mappings, types
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
            mappings, types
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
            mappings, types
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
            mappings, types
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
            mappings, types
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
            mappings, types
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
            mappings, types
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
            mappings, types
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

    def translate(self):
        return z3.Not(self.cond.translate())

    def eval(self, x):
        return not self.cond.eval(x)

    def __str__(self):
        return f"!({self.cond})"

    def collect_vars(self):
        return self.cond.collect_vars()
