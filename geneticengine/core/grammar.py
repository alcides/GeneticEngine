from __future__ import annotations

from abc import ABC
from abc import ABCMeta
from collections import defaultdict
from inspect import isclass
from tracemalloc import start
from typing import Any, Generic
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from typing import Type

from geneticengine.core.utils import get_arguments
from geneticengine.core.utils import get_generic_parameter
from geneticengine.core.utils import get_generic_parameters
from geneticengine.core.utils import is_abstract
from geneticengine.core.utils import is_annotated
from geneticengine.core.utils import is_generic
from geneticengine.core.utils import is_generic_list
from geneticengine.core.utils import is_terminal
from geneticengine.core.utils import strip_annotations


class Grammar:
    starting_symbol: type
    alternatives: dict[type, list[type]]
    distanceToTerminal: dict[Any, int]
    all_nodes: set[type]
    recursive_prods: set[type]
    terminals: set[
        type
    ]  # todo: both terminals and non_terminals can be obtained by checking if disttoterminal == or!= 0
    non_terminals: set[type]
    abstract_dist_to_t: dict[type, dict[type, int]]

    def __init__(
        self,
        starting_symbol: type,
        considered_subtypes: list[type] = None,
    ) -> None:
        self.alternatives: dict[type, list[type]] = {}
        self.starting_symbol = starting_symbol
        self.distanceToTerminal = {int: 1, str: 1, float: 1}
        self.all_nodes = set()
        self.recursive_prods = set()
        self.terminals = set()
        self.non_terminals = set()
        self.abstract_dist_to_t = defaultdict(
            lambda: defaultdict(lambda: 1000000),
        )
        self.considered_subtypes = considered_subtypes or []

    def register_alternative(self, nonterminal: type, nodetype: type):
        """
        Register a production A->B
        Call multiple times with same A to register many possible alternatives.
        """
        if not is_abstract(nonterminal):
            raise Exception(
                f"Trying to register an alternative on a non-abstract class: {nonterminal} -> {nodetype}\n"
                f"You may have meant for {nonterminal.__name__} to be abstract. If so, decorate it with @abstract.\n"
                f"(Found in geneticengine.core.decorators)"
            )
        if nonterminal not in self.alternatives:
            self.alternatives[nonterminal] = []
        self.alternatives[nonterminal].append(nodetype)

    def register_type(self, ty: type):
        if ty in self.all_nodes:
            return
        elif is_generic_list(ty):
            gty = get_generic_parameter(ty)
            self.register_type(gty)
            return
        elif is_annotated(ty):
            gty = get_generic_parameter(ty)
            self.register_type(gty)
            return
        elif is_generic(ty):
            for p in get_generic_parameters(ty):
                self.register_type(p)
            return
        self.all_nodes.add(ty)

        parent = ty.mro()[1]
        if parent not in [object, ABC, Generic]:
            assert isinstance(parent, type)
            self.register_type(parent)
            self.register_alternative(parent, ty)

        terminal = False
        if not is_abstract(ty):
            terminal = True
            for (arg, argt) in get_arguments(ty):
                terminal = False
                if isinstance(argt, type) or isinstance(argt, ABCMeta):
                    self.register_type(argt)

        for st in self.considered_subtypes:
            if issubclass(st, ty):
                self.register_type(st)

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
            args = ", ".join(
                [f"{a}: {wrap(at)}" for (a, at) in get_arguments(x)],
            )
            return f"{x.__name__}({args})"

        prods = ";".join(
            [
                str(p.__name__)
                + " -> "
                + ("|".join([format(p) for p in self.alternatives[p]]))
                for p in self.alternatives
            ],
        )
        return (
            f"Grammar<Starting={self.starting_symbol.__name__},Productions=[{prods}]>"
        )

    def get_all_symbols(self) -> tuple[set[type], set[type], set[type]]:
        """All symbols in the current grammar, including terminals"""
        keys = {k for k in self.alternatives.keys()}
        sequence = {v for vv in self.alternatives.values() for v in vv}
        return (keys, sequence, sequence.union(keys).union(self.all_nodes))

    def get_distance_to_terminal(self, ty: type) -> int:
        """Returns the current distance to terminal of a given type"""
        if is_annotated(ty):
            ta = get_generic_parameter(ty)
            return self.get_distance_to_terminal(ta)
        elif is_generic_list(ty):
            ta = get_generic_parameter(ty)
            return 1 + self.get_distance_to_terminal(ta)
        elif is_generic(ty):
            return 1 + max(
                self.get_distance_to_terminal(t) for t in get_generic_parameters(ty)
            )
        else:
            return self.distanceToTerminal[ty]

    def get_min_tree_depth(self):
        """Returns the minimum depth a tree must have"""
        return self.distanceToTerminal[self.starting_symbol]

    def get_max_node_depth(self):
        """Returns the maximum minimum depth a node can have"""

        def dist(x):
            return self.distanceToTerminal[x]

        return max(list(map(dist, self.all_nodes)))

    def preprocess(self):
        """Computes distanceToTerminal via a fixpoint algorithm."""
        (keys, _, all_sym) = self.get_all_symbols()
        for s in all_sym:
            self.distanceToTerminal[s] = 1000000
        changed = True

        reachability: dict[type, set[type]] = defaultdict(lambda: set())

        def process_reachability(src: type, dsts: list[type]):
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
                            val = min(val, 1 + self.distanceToTerminal[prod])
                            old = self.abstract_dist_to_t[sym][prod]
                            if 1 < old:
                                self.abstract_dist_to_t[sym][prod] = 1
                                changed = True
                            if prod in self.abstract_dist_to_t:
                                for subprod, dist in self.abstract_dist_to_t[
                                    prod
                                ].items():
                                    currdist = self.abstract_dist_to_t[sym][subprod]
                                    candidate = dist + 1
                                    if candidate < currdist:
                                        self.abstract_dist_to_t[sym][
                                            subprod
                                        ] = candidate
                                        changed = True
                        changed |= process_reachability(sym, prods)
                else:
                    if is_terminal(sym, self.non_terminals):
                        val = 1
                    else:
                        args = get_arguments(sym)
                        assert args
                        val = max(
                            1 + self.get_distance_to_terminal(argt)
                            for (_, argt) in args
                        )

                        changed |= process_reachability(
                            sym,
                            [argt for (_, argt) in args],
                        )

                if val < old_val:
                    changed = True
                    self.distanceToTerminal[sym] = val

        for sym in all_sym:
            if sym in reachability[sym]:  # symbol is recursive
                self.recursive_prods.add(sym)
            else:
                pass


def extract_grammar(
    considered_subtypes: list[type],
    starting_symbol: type,
):
    """
    The extract_grammar takes in all the productions of the grammar (nodes) and a starting symbol (starting_symbol). It goes through all the nodes and constructs a complete grammar that can then be used for search algorithms such as Genetic Programming and Hill Climbing.

    Parameters:
        - nodes (list): A list of objects representing tree nodes. Make sure that any node can be produced be the starting symbol.
        - starting_symbol (object): The starting symbol of each tree. Makes sure every generated tree by the returned grammar starts with this symbol. Make sure that the starting symbol can produce any object of nodes.

    Returns:
        - The grammar

    """
    g = Grammar(starting_symbol, considered_subtypes)
    g.register_type(starting_symbol)
    g.preprocess()
    return g
