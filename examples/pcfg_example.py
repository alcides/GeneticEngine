from __future__ import annotations

from abc import ABC

from geml.simplegp import SimpleGP
from geneticengine.grammar.decorators import weight
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.solutions.tree import TreeNode

# ===================================
# This is an example of how to create Probabilistic grammars.
# In this example, we assign weights to the grammar to create Probabilistic grammars.
# ===================================


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


def fitness_function(x) -> float:
    return 1.0


if __name__ == "__main__":
    g = extract_grammar([A, B, C], R)

    alg = SimpleGP(
        grammar=g,
        representation="treebased",
        fitness_function=fitness_function,
        minimize=False,
        max_depth=10,
        population_size=1000,
        max_evaluations=10 * 1000,
    )
    ind = alg.search()[0]

    x: TreeNode = ind.get_phenotype()
    print(x)
