from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated


from geneticengine.grammar.grammar import extract_grammar
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.treebased import TreeBasedRepresentation, random_node
from geneticengine.grammar.metahandlers.lists import ListSizeBetween


@dataclass
class Root:
    l: Annotated[list[B], ListSizeBetween(1, 6)]


@dataclass
class B:
    i: int


class TestTreeBased:
    def test_mutation_empty_list(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([Root, B], Root)
        tbr = TreeBasedRepresentation(g, max_depth=3)
        t = random_node(r, g, max_depth=3)
        # t = tbr.instantiate(r)
        for i in range(1000):
            t = tbr.mutate(r, t)
        assert len(t.l) >= 0
