from dataclasses import dataclass
from typing import Annotated, Any, Callable

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource, RandomSource
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammar.metahandlers.ints import IntRange


@dataclass
class Dependent(MetaHandlerGenerator):
    name: str
    callable: Callable[[Any], type]

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        t: type = self.callable(dependent_values[self.name])
        return rec(Annotated[base_type, t])

    def __hash__(self):
        return hash(self.__class__) + hash(self.name) + hash(id(self.callable))


@dataclass
class SimplePair:
    a: Annotated[int, IntRange(0, 100)]
    b: Annotated[int, Dependent("a", lambda a: IntRange(a, 102))]


def test_dependent_types():
    r = NativeRandomSource(seed=1)
    g = extract_grammar([SimplePair], SimplePair)
    max_depth = 3

    repr = TreeBasedRepresentation(g, max_depth)

    def fitness_function(x: SimplePair) -> float:
        return x.b - x.a

    for _ in range(100):
        gp = GeneticProgramming(
            representation=repr,
            problem=SingleObjectiveProblem(fitness_function=fitness_function, minimize=True),
            random=r,
            budget=EvaluationBudget(1000),
        )
        ind = gp.search()
        p = ind.get_phenotype()
        assert p.a <= p.b


def test_dependent_types_mutation():
    r = NativeRandomSource(seed=1)
    g = extract_grammar([SimplePair], SimplePair)
    max_depth = 3

    repr = TreeBasedRepresentation(g, max_depth)

    el = repr.create_genotype(r)
    for _ in range(1000):
        p = repr.mutate(r, el)
        print(p)
        assert p.a <= p.b
    assert False
