from dataclasses import dataclass
import os
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.core.representations.grammatical_evolution import ge_representation
from geneticengine.algorithms.gp.gp import GP
from geneticengine.grammars.coding.control_flow import ForLoop, Code
from geneticengine.grammars.coding.classes import Expr, Statement, XAssign


class VarX(Expr):
    def evaluate(self, x=0):
        return x

    def __str__(self) -> str:
        return "x"


class Const(Expr):
    def evaluate(self, x=0):
        return 0.5

    def __str__(self) -> str:
        return "0.5"


@dataclass
class XPlusConst(Expr):
    right: Const

    def evaluate(self, x):
        return x + self.right.evaluate(x)

    def __str__(self) -> str:
        return "x + {}".format(self.right)


@dataclass
class XTimesConst(Expr):
    right: Const

    def evaluate(self, x):
        return x * self.right.evaluate(x)

    def __str__(self) -> str:
        return "x * {}".format(self.right)


def fit(indiv: Code):
    return indiv.evaluate()


fitness_function = lambda x: fit(x)


def preprocess():
    return extract_grammar(
        [XPlusConst, XTimesConst, XAssign, ForLoop, Code, Const, VarX], ForLoop
    )


def evolve(g, seed, mode, representation='treebased_representation', output_folder=('','all')):
    if representation == 'grammatical_evolution':
        representation = ge_representation
    else:
        representation = treebased_representation
    
    alg = GP(
        g,
        fitness_function,
        representation=representation,
        # max_depth=13,
        # population_size=1,
        # number_of_generations=1,
        minimize=False,
        seed=seed,
        timer_stop_criteria=mode,
        safe_gen_to_csv=output_folder
    )
    (b, bf, bp) = alg.evolve(verbose=2)
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
