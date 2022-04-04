from __future__ import annotations

from abc import ABC
from typing import Annotated
from typing import List
from typing import NamedTuple
from typing import Protocol

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.decorators import weight
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween


class R(ABC):
    pass


@weight(0.1)
class A(R):
    pass


@weight(0.8)
class B(R):
    pass


@weight(0.1)
class C(R):
    pass


if __name__ == "__main__":
    g = extract_grammar([A, B, C], R)
    alg = GP(
        g,
        lambda x: 1,
        representation=treebased_representation,
        max_depth=10,
        population_size=1000,
        number_of_generations=1,
        minimize=False,
    )
    (b, bf, bp) = alg.evolve(verbose=1)

    def count(xs):
        d = {}
        for x in xs:
            if x not in d:
                d[x] = 1
            else:
                d[x] += 1
        return d

    print(count([x.genotype.__class__ for x in alg.final_population]))
