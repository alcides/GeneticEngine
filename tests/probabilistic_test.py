from __future__ import annotations

from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.core.decorators import abstract
from geneticengine.core.decorators import weight
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.ge import (
    GrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation


@abstract
class Option:
    pass


@weight(1)
class OptionA(Option):
    pass


@weight(99)
class OptionB(Option):
    pass


class TestProbabilisticGrammar:
    def test_probabilistic_grammar_tree_based(self):
        g = extract_grammar([OptionA, OptionB], Option)

        gp = GP(
            representation=TreeBasedRepresentation(grammar=g, max_depth=10),
            problem=SingleObjectiveProblem(
                lambda p: isinstance(p, OptionA) and 1 or 2,
                minimize=True,
                target_fitness=0,
            ),
            population_size=1000,
            stopping_criterium=GenerationStoppingCriterium(max_generations=50),
        )
        a, b, c = gp.evolve()
        assert isinstance(c, OptionA)

    def test_probabilistic_grammar_ge(self):
        g = extract_grammar([OptionA, OptionB], Option)

        gp = GP(
            representation=GrammaticalEvolutionRepresentation(grammar=g, max_depth=10),
            problem=SingleObjectiveProblem(
                lambda p: isinstance(p, OptionA) and 1 or 2,
                minimize=True,
                target_fitness=0,
            ),
            population_size=1000,
            stopping_criterium=GenerationStoppingCriterium(max_generations=50),
        )
        a, b, c = gp.evolve()
        assert isinstance(c, OptionA)
