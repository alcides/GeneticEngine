from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Generic,
    Protocol,
    Type,
    TypeVar,
    Tuple,
    List,
    _AnnotatedAlias,
    _GenericAlias,
    Union,
    cast,
)

from geneticengine.core.utils import get_arguments


class Grammar(object):
    starting_symbol: type
    productions: Dict[type, List[type]]
    distanceToTerminal: Dict[Any, int]

    def __init__(self, starting_symbol) -> None:
        self.productions: Dict[type, List[type]] = {}
        self.starting_symbol = starting_symbol
        self.distanceToTerminal = {int: 1, str: 1, float: 1}

    def register(self, nonterminal: type, nodetype: type):
        if nonterminal not in self.productions:
            self.productions[nonterminal] = []
        self.productions[nonterminal].append(nodetype)

    def extract(self, ty: type, nodes: List[type]):
        if ty in self.productions.keys():
            return
        for n in nodes:
            if ty in n.mro():
                self.register(ty, n)
                for (arg, argt) in get_arguments(n):
                    self.extract(argt, nodes)

    def __repr__(self):
        def wrap(n):
            if hasattr(n, "__name__"):
                return n.__name__
            if hasattr(n, "__metadata__"):
                return n.__metadata__[0]
            return n

        def format(x):
            args = ", ".join([f"{a}: {wrap(at)}" for (a, at) in get_arguments(x)])
            return f"{x.__name__}({args})"

        prods = ";".join(
            [
                str(p.__name__)
                + " -> "
                + ("|".join([format(p) for p in self.productions[p]]))
                for p in self.productions
            ]
        )
        return (
            f"Grammar<Starting={self.starting_symbol.__name__},Productions=[{prods}]>"
        )

    def preprocess(self):
        choice = set()
        for k in self.productions.keys():
            choice.add(k)
        sequence = set()
        for vv in self.productions.values():
            for v in vv:
                if v not in choice:
                    sequence.add(v)
        all_sym = sequence.union(choice)
        for s in all_sym:
            self.distanceToTerminal[s] = 1000000
        changed = True
        while changed:
            changed = False
            for sym in all_sym:
                old_val = self.distanceToTerminal[sym]
                val = old_val
                if sym in choice:
                    for prod in self.productions[sym]:
                        val = min(val, self.distanceToTerminal[prod])
                else:
                    if hasattr(sym, "__annotations__"):
                        var = sym.__annotations__.values()
                        if isinstance(list(var)[0], _AnnotatedAlias):
                            t = list(var)[0].__origin__
                        else:
                            t = var.__iter__().__next__()
                        if isinstance(t, _GenericAlias):
                            t = t.__args__[0]
                        val = self.distanceToTerminal[t]
                        for prod in var:
                            if isinstance(prod, _AnnotatedAlias):
                                prod = prod.__origin__
                            if isinstance(prod, _GenericAlias):
                                prod = prod.__args__[0]
                            val = max(val, self.distanceToTerminal[prod] + 1)
                    else:
                        val = 1
                if val != old_val:
                    changed = True
                    self.distanceToTerminal[sym] = val


def extract_grammar(nodes, starting_symbol):
    g = Grammar(starting_symbol)
    g.extract(starting_symbol, nodes)
    g.preprocess()
    return g
