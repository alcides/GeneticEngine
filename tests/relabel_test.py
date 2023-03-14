from __future__ import annotations

from dataclasses import dataclass

from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.utils import relabel_nodes_of_trees


@abstract
class Root:
    pass


@dataclass
class Middle(Root):
    child: Root


class Concrete(Root):
    pass


class TestRelabel:
    def test_relabel_simple(self):
        c = Concrete()
        g = extract_grammar([Concrete], Root)
        relabel_nodes_of_trees(c, g)
        assert c.gengy_distance_to_term == 0
