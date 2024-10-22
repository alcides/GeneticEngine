from dataclasses import dataclass
from typing import Annotated

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.metahandlers.dependent import Dependent
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammar.metahandlers.ints import IntRange


@dataclass
class SimplePair:
    a: Annotated[int, IntRange(0, 3)]
    b: Annotated[int, Dependent("a", lambda a: IntRange(a, 4))]
    c: int


def test_dependent_types():
    r = NativeRandomSource(seed=1)
    g = extract_grammar([SimplePair], SimplePair)
    decider = MaxDepthDecider(r, g, 3)

    repr = TreeBasedRepresentation(g, decider)

    def fitness_function(x: SimplePair) -> float:
        return x.b - x.a

    for _ in range(100):
        gp = GeneticProgramming(
            representation=repr,
            problem=SingleObjectiveProblem(fitness_function=fitness_function, minimize=True),
            random=r,
            budget=EvaluationBudget(10),
        )
        ind = gp.search()[0]
        p = ind.get_phenotype()
        assert p.a <= p.b


def test_dependent_types_mutation():
    r = NativeRandomSource(seed=1)
    g = extract_grammar([SimplePair], SimplePair)
    decider = MaxDepthDecider(r, g, 3)

    repr = TreeBasedRepresentation(g, decider)

    for _ in range(10):
        el = repr.create_genotype(r)
        for _ in range(10):
            p = repr.mutate(r, el)
            print(p)
            assert p.a <= p.b
