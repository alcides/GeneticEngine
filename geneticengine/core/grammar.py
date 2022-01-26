from abc import ABC
from collections import defaultdict
from typing import Any, Dict, List, Set, Type, Tuple

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
    alternatives: Dict[type, List[type]]
    distanceToTerminal: Dict[Any, int]
    all_nodes: Set[type]
    recursive_prods: Set[type]
    terminals: Set[
        type
    ]  # todo: both terminals and non_terminals can be obtained by checking if disttoterminal == or!= 0
    non_terminals: Set[type]

    def __init__(self, starting_symbol) -> None:
        self.alternatives: Dict[type, List[type]] = {}
        self.starting_symbol = starting_symbol
        self.distanceToTerminal = {int: 1, str: 1, float: 1}
        self.all_nodes = set()
        self.recursive_prods = set()
        self.terminals = set()
        self.non_terminals = set()

    def register_alternative(self, nonterminal: type, nodetype: type):
        """
        Register a production A->B
        Call multiple times with same A to register many possible alternatives.
        """
        if nonterminal not in self.alternatives:
            self.alternatives[nonterminal] = []
        self.alternatives[nonterminal].append(nodetype)

    def register_type(self, ty: type):
        if ty in self.all_nodes:
            return
        elif is_generic_list(ty) or is_annotated(ty):
            self.register_type(get_generic_parameter(ty))
            return
        self.all_nodes.add(ty)

        parent = ty.mro()[1]
        if parent not in [object, ABC]:
            self.register_type(parent)
            self.register_alternative(parent, ty)

        terminal = False
        if not is_abstract(ty):
            terminal = True
            for (arg, argt) in get_arguments(ty):
                terminal = False
                self.register_type(argt)

        if terminal:
            self.terminals.add(ty)
        else:
            self.non_terminals.add(ty)

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
                + ("|".join([format(p) for p in self.alternatives[p]]))
                for p in self.alternatives
            ]
        )
        return (
            f"Grammar<Starting={self.starting_symbol.__name__},Productions=[{prods}]>"
        )

    def get_all_symbols(self) -> Tuple[Set[Type], Set[Type], Set[Type]]:
        """All symbols in the current grammar, including terminals"""
        keys = set((k for k in self.alternatives.keys()))
        sequence = set((v for vv in self.alternatives.values() for v in vv))
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
        return max(list(map(dist, self.all_nodes)))

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
                    if sym in self.alternatives:
                        prods = self.alternatives[sym]
                        for prod in prods:
                            val = min(
                                val, self.distanceToTerminal[prod]
                            )  # todo: No +1 because alternatives don't
                            # todo: actually take tree space. This is problematic and can lead to infinite loops, but
                            # todo: our implementation is safe because a object cannot inherit itself in a cycle
                            # todo: we may want to revisit this

                        changed |= process_reachability(sym, prods)
                else:
                    if is_terminal(sym, self.non_terminals):
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
            if sym in reachability[sym]:  # symbol is recursive
                self.recursive_prods.add(sym)
            else:
                pass


def extract_grammar(nodes, starting_symbol):
    g = Grammar(starting_symbol)
    g.register_type(starting_symbol)
    for n in nodes:
        g.register_type(n)
    g.preprocess()
    return g
