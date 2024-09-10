from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated


from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.representations.tree.utils import relabel_nodes_of_trees
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
