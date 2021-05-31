from typing import Dict, List

from geneticengine.core.utils import get_arguments


class Grammar(object):
    def __init__(self, starting_symbol) -> None:
        self.productions: Dict[type, List[type]] = {}
        self.starting_symbol = starting_symbol

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


def extract_grammar(nodes, starting_symbol):
    g = Grammar(starting_symbol)
    g.extract(starting_symbol, nodes)
    return g
