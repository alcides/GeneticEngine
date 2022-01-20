from dataclasses import dataclass

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation


@abstract
@dataclass
class Root:
    pass


@abstract
@dataclass
class A(Root):
    pass


@dataclass
class B(A):
    inst: A


@dataclass
class C(A):
    pass


if __name__ == "__main__":
    g = extract_grammar(
        [
            A,
            B,
            C,
        ],
        Root,
    )
    alg = GP(
        g,
        lambda x: 0,
        representation=treebased_representation,
        max_depth=5,
        population_size=1000,
        n_elites=100,
        number_of_generations=100,
        probability_crossover=0.5,
        selection_method=("tournament", 2),
        minimize=True,
    )
    print("Started running...")
    (b, bf, bp) = alg.evolve(verbose=1)
    print(f"Best individual: {b}")
    print(f"With fitness: {bf}")
