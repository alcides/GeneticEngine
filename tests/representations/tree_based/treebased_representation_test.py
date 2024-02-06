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
        print(g)
        print(g.get_grammar_properties_summary())
        tbr = TreeBasedRepresentation(g, max_depth=3)
        mutation = tbr.get_mutation()
        t = random_node(r, g, max_depth=3)
        t = tbr.create_individual(r)
        for i in range(1000):
            t = mutation.mutate(t, None, None, tbr, r, 0, i)
        assert len(t.l) >= 0
