from __future__ import annotations

import pytest

from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.stop import (
    AnyOfStoppingCriterium,
    SingleFitnessTargetStoppingCriterium,
    GenerationStoppingCriterium,
)
from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.decorators import weight
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.representations.grammatical_evolution.dynamic_structured_ge import (
    DynamicStructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.representations.grammatical_evolution.ge import (
    GrammaticalEvolutionRepresentation,
)
from geneticengine.representations.grammatical_evolution.structured_ge import (
    StructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


@abstract
class Option:
    pass


@weight(1)
class OptionA(Option):
    pass


@weight(99)
class OptionB(Option):
    pass


@pytest.mark.skip(reason="Takes too long")
class TestProbabilisticGrammar:
    def test_probabilistic_grammar_tree_based(self):
        g = extract_grammar([OptionA, OptionB], Option)

        gp = GP(
            representation=TreeBasedRepresentation(grammar=g, max_depth=10),
            problem=SingleObjectiveProblem(
                lambda p: isinstance(p, OptionA) and 1 or 2,
                minimize=True,
            ),
            population_size=1000,
            stopping_criterium=GenerationStoppingCriterium(max_generations=50),
        )
        ind = gp.evolve()
        tree = ind.get_phenotype()
        assert isinstance(tree, OptionA)

    def test_probabilistic_grammar_ge(self):
        g = extract_grammar([OptionA, OptionB], Option)

        stopping_criterium = AnyOfStoppingCriterium(
            GenerationStoppingCriterium(max_generations=50),
            SingleFitnessTargetStoppingCriterium(0),
        )
        gp = GP(
            representation=GrammaticalEvolutionRepresentation(grammar=g, max_depth=10),
            problem=SingleObjectiveProblem(
                lambda p: isinstance(p, OptionA) and 1 or 2,
                minimize=True,
            ),
            population_size=1000,
            stopping_criterium=stopping_criterium,
        )
        ind = gp.evolve()
        tree = ind.get_phenotype()
        assert isinstance(tree, OptionA)

    def test_probabilistic_grammar_sge(self):
        g = extract_grammar([OptionA, OptionB], Option)

        gp = GP(
            representation=StructuredGrammaticalEvolutionRepresentation(
                grammar=g,
                max_depth=10,
            ),
            problem=SingleObjectiveProblem(
                lambda p: isinstance(p, OptionA) and 1 or 2,
                minimize=True,
            ),
            population_size=1000,
            stopping_criterium=AnyOfStoppingCriterium(
                GenerationStoppingCriterium(max_generations=50),
                SingleFitnessTargetStoppingCriterium(0),
            ),
        )
        ind = gp.evolve()
        tree = ind.get_phenotype()
        assert isinstance(tree, OptionA)

    def test_probabilistic_grammar_dsge(self):
        g = extract_grammar([OptionA, OptionB], Option)

        gp = GP(
            representation=DynamicStructuredGrammaticalEvolutionRepresentation(
                grammar=g,
                max_depth=10,
            ),
            problem=SingleObjectiveProblem(
                lambda p: isinstance(p, OptionA) and 1 or 2,
                minimize=True,
            ),
            population_size=1000,
            stopping_criterium=AnyOfStoppingCriterium(
                GenerationStoppingCriterium(max_generations=50),
                SingleFitnessTargetStoppingCriterium(0),
            ),
        )
        ind = gp.evolve()
        tree = ind.get_phenotype()
        assert isinstance(tree, OptionA)
