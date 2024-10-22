from __future__ import annotations


from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.decorators import weight
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.grammatical_evolution.dynamic_structured_ge import (
    DynamicStructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.representations.grammatical_evolution.structured_ge import (
    StructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.representations.tree.initializations import MaxDepthDecider
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


class TestProbabilisticGrammar:
    def test_probabilistic_grammar_tree_based(self):
        g = extract_grammar([OptionA, OptionB], Option)
        r = NativeRandomSource(0)

        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(grammar=g, decider=MaxDepthDecider(r, g, 10)),
            problem=SingleObjectiveProblem(
                lambda p: isinstance(p, OptionA) and 1 or 2,
                minimize=True,
            ),
            population_size=1000,
            budget=EvaluationBudget(50 * 1000),
        )
        ind = gp.search()[0]
        tree = ind.get_phenotype()
        assert isinstance(tree, OptionA)

    def test_probabilistic_grammar_ge(self):
        g = extract_grammar([OptionA, OptionB], Option)
        r = NativeRandomSource(0)

        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(grammar=g, decider=MaxDepthDecider(r, g, 10)),
            problem=SingleObjectiveProblem(
                lambda p: isinstance(p, OptionA) and 1 or 2,
                minimize=True,
                target=0,
            ),
            population_size=1000,
            budget=EvaluationBudget(50 * 1000),
        )
        ind = gp.search()[0]
        tree = ind.get_phenotype()
        assert isinstance(tree, OptionA)

    def test_probabilistic_grammar_sge(self):
        g = extract_grammar([OptionA, OptionB], Option)
        r = NativeRandomSource(0)

        gp = GeneticProgramming(
            representation=StructuredGrammaticalEvolutionRepresentation(grammar=g, decider=MaxDepthDecider(r, g, 10)),
            problem=SingleObjectiveProblem(
                lambda p: isinstance(p, OptionA) and 1 or 2,
                minimize=True,
                target=0,
            ),
            population_size=1000,
            budget=EvaluationBudget(50 * 1000),
        )
        ind = gp.search()[0]
        tree = ind.get_phenotype()
        assert isinstance(tree, OptionA)

    def test_probabilistic_grammar_dsge(self):
        g = extract_grammar([OptionA, OptionB], Option)
        r = NativeRandomSource(1)

        gp = GeneticProgramming(
            problem=SingleObjectiveProblem(
                lambda p: isinstance(p, OptionA) and 1 or 2,
                minimize=True,
                target=0,
            ),
            budget=EvaluationBudget(50 * 1000),
            representation=DynamicStructuredGrammaticalEvolutionRepresentation(grammar=g, max_depth=10),
            random=r,
            population_size=1000,
        )
        ind = gp.search()[0]
        tree = ind.get_phenotype()
        assert isinstance(tree, OptionA)
