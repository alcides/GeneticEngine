from collections import defaultdict
from typing import (
    Any,
    Dict,
    List,
    Set,
)

from geneticengine.core.utils import (
    get_arguments,
    get_generic_parameter,
    is_abstract,
    is_annotated,
    is_generic_list,
    is_terminal,
    strip_annotations,
)


class Grammar(object):
    starting_symbol: type
    productions: Dict[type, List[type]]
    distanceToTerminal: Dict[Any, int]
    nodes: list[type]
    recursive_prods: Set[type]

    def __init__(self, starting_symbol, nodes) -> None:
        self.productions: Dict[type, List[type]] = {}
        self.starting_symbol = starting_symbol
        self.distanceToTerminal = {int: 1, str: 1, float: 1}
        self.nodes = nodes
        self.recursive_prods = set()

    def non_terminals(self) -> list[type]:
        return self.nodes

    def register(self, nonterminal: type, nodetype: type):
        if nonterminal not in self.productions:
            self.productions[nonterminal] = []
        self.productions[nonterminal].append(nodetype)

    def extract(self, ty: type):
        if ty in self.productions.keys():
            return
        elif is_generic_list(ty) or is_annotated(ty):
            self.extract(get_generic_parameter(ty))
        else:
            for n in self.nodes:
                if ty in n.mro():
                    self.register(ty, n)
                    for (arg, argt) in get_arguments(n):
                        self.extract(argt)

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

    def get_all_symbols(self) -> tuple[set[type], set[type], set[type]]:
        """All symbols in the current grammar, including terminals"""
        keys = set((k for k in self.productions.keys()))
        sequence = set((v for vv in self.productions.values() for v in vv))
        return (keys, sequence, sequence.union(keys))

    def get_distance_to_terminal(self, ty: type) -> int:
        """Returns the current distance to terminal of a given type"""
        if is_generic_list(ty) or is_annotated(ty):
            ta = get_generic_parameter(ty)
            return 1 + self.get_distance_to_terminal(ta)
        return self.distanceToTerminal[ty]

    def get_min_tree_depth(self):
        """Returns the minimum depth a tree must have"""
        return self.distanceToTerminal[self.starting_symbol]

    def get_max_node_depth(self):
        """Returns the maximum minimum depth a node can have"""
        dist = lambda x: self.distanceToTerminal[x]
        return max(list(map(dist, self.nodes)))

    def preprocess(self):
        """Computes distanceToTerminal via a fixpoint algorithm."""
        (keys, _, all_sym) = self.get_all_symbols()
        for s in all_sym:
            self.distanceToTerminal[s] = 1000000
        changed = True

        reachability: Dict[type, Set[type]] = defaultdict(lambda: set())

        def process_reachability(src: type, dsts: List[type]):
            src = strip_annotations(src)
            ch = False
            src_reach = reachability[src]
            for prod in dsts:
                prod = strip_annotations(prod)
                reach = reachability[prod]
                oldlen = len(reach)
                reach.add(src)
                reach.update(src_reach)
                ch |= len(reach) != oldlen
            return ch

        while changed:
            changed = False
            for sym in all_sym:
                old_val = self.distanceToTerminal[sym]
                val = old_val

                if is_abstract(sym):
                    if sym in self.productions:
                        prods = self.productions[sym]
                        for prod in prods:
                            val = min(val, self.distanceToTerminal[prod])

                        changed |= process_reachability(sym, prods)
                else:
                    if is_terminal(sym, self.non_terminals()):
                        val = 1
                    else:
                        args = get_arguments(sym)
                        assert args
                        val = 1 + max(
                            [self.get_distance_to_terminal(argt) for (_, argt) in args]
                        )

                        changed |= process_reachability(
                            sym, [argt for (_, argt) in args]
                        )

                if val < old_val:
                    changed = True
                    self.distanceToTerminal[sym] = val

        for sym in all_sym:
            # print(sym.__name__)
            if sym in reachability[sym]:  # symbol is recursive
                self.recursive_prods.add(sym)
                # print("yes")
            else:
                # print("no")
                pass


def extract_grammar(nodes, starting_symbol):
    g = Grammar(starting_symbol, nodes)
    g.extract(starting_symbol)
    g.preprocess()
    return g
