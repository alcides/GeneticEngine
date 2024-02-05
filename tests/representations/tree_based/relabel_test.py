from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.random.sources import RandomSource
from geneticengine.representations.tree.initializations import grow_method
from geneticengine.representations.tree.treebased import random_individual
from geneticengine.representations.tree.utils import get_nodes_depth_specific, relabel_nodes_of_trees
from geneticengine.grammar.metahandlers.lists import ListSizeBetween


@abstract
class Root:
    pass


@dataclass
class Middle(Root):
    child: Root


class Concrete(Root):
    pass


@dataclass
class MiddleList(Root):
    z: Annotated[list[Root], ListSizeBetween(2, 3)]


class TestRelabel:
    def test_relabel_simple(self):
        c = Concrete()
        g = extract_grammar([Concrete], Root)
        relabel_nodes_of_trees(c, g)
        assert c.gengy_distance_to_term == 0


class TestNodesDepthSpecific:
    def test_nodes_depth_specific_simple(self):
        r = RandomSource(123)
        g1 = extract_grammar([Concrete, Middle, MiddleList], Middle)
        g2 = extract_grammar([Concrete, Middle, MiddleList], MiddleList)
        g3 = extract_grammar([Concrete, Middle], Middle)

        def find_depth_specific_nodes(r, g, depth):
            ind = random_individual(r, g, depth, method=grow_method)
            return get_nodes_depth_specific(ind, g)

        branching_average1 = g1.get_branching_average_proxy(r, find_depth_specific_nodes, 100, 17)
        assert branching_average1["0"] == 1
        branching_average2 = g2.get_branching_average_proxy(r, find_depth_specific_nodes, 100, 17)
        assert branching_average2["0"] >= 2 and branching_average2["0"] <= 3
        branching_average3 = g3.get_branching_average_proxy(r, find_depth_specific_nodes, 100, 17)
        assert branching_average3["0"] == branching_average3["1"]
