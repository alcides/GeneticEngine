from __future__ import annotations

from abc import ABC

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.decorators import weight
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.tree.treebased import treebased_representation

"""
This is a simple example on how to use GeneticEngine to solve a GP problem.
We define the tree structure of the representation and we assigned weights to each tree node,
then we define the fitness function for our problem.
"""
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
        representation=treebased_representation,
        problem=SingleObjectiveProblem(
            minimize=False,
            fitness_function=lambda x: 1,
            target_fitness=None,
        ),
        max_depth=10,
        population_size=1000,
        number_of_generations=10,
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
