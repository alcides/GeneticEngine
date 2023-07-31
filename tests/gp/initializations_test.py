from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.initialization_methods import (
    FullInitializer,
    GrowInitializer,
    PositionIndependentGrowInitializer,
    RampedHalfAndHalfInitializer,
)
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.ge import GrammaticalEvolutionRepresentation
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass
class Rec(Root):
    a: Root
    b: Root


@dataclass
class Option(Root):
    a: int


def fitness_function(r: Root) -> float:
    return 1


class TestPIGrow:
    def test_treebased(self):
        population_size = 11

        grammar = extract_grammar([Option, Rec], Root)
        gp = GP(
            representation=TreeBasedRepresentation(grammar=grammar, max_depth=5),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            stopping_criterium=GenerationStoppingCriterium(2),
            population_size=population_size,
            initializer=PositionIndependentGrowInitializer(),
        )
        gp.evolve()

        assert gp.final_population

    def test_ge(self):
        population_size = 11

        grammar = extract_grammar([Option, Rec], Root)
        gp = GP(
            representation=GrammaticalEvolutionRepresentation(grammar=grammar, max_depth=5),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            stopping_criterium=GenerationStoppingCriterium(2),
            population_size=population_size,
            initializer=PositionIndependentGrowInitializer(),
        )
        gp.evolve()

        assert gp.final_population


class TestGrow:
    def test_treebased(self):
        population_size = 11

        grammar = extract_grammar([Option, Rec], Root)
        gp = GP(
            representation=TreeBasedRepresentation(grammar=grammar, max_depth=5),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            stopping_criterium=GenerationStoppingCriterium(2),
            population_size=population_size,
            initializer=GrowInitializer(),
        )
        gp.evolve()

        assert gp.final_population

    def test_ge(self):
        population_size = 11

        grammar = extract_grammar([Option, Rec], Root)
        gp = GP(
            representation=GrammaticalEvolutionRepresentation(grammar=grammar, max_depth=5),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            stopping_criterium=GenerationStoppingCriterium(2),
            population_size=population_size,
            initializer=GrowInitializer(),
        )
        gp.evolve()

        assert gp.final_population


class TestFull:
    def test_treebased(self):
        population_size = 11

        grammar = extract_grammar([Option, Rec], Root)
        gp = GP(
            representation=TreeBasedRepresentation(grammar=grammar, max_depth=5),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            stopping_criterium=GenerationStoppingCriterium(2),
            population_size=population_size,
            initializer=FullInitializer(),
        )
        gp.evolve()

        assert gp.final_population

    def test_ge(self):
        population_size = 11

        grammar = extract_grammar([Option, Rec], Root)
        gp = GP(
            representation=GrammaticalEvolutionRepresentation(grammar=grammar, max_depth=5),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            stopping_criterium=GenerationStoppingCriterium(2),
            population_size=population_size,
            initializer=FullInitializer(),
        )
        gp.evolve()


class TestRampedHAH:
    def test_treebased(self):
        population_size = 11

        grammar = extract_grammar([Option, Rec], Root)
        gp = GP(
            representation=TreeBasedRepresentation(grammar=grammar, max_depth=5),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            stopping_criterium=GenerationStoppingCriterium(2),
            population_size=population_size,
            initializer=RampedHalfAndHalfInitializer(),
        )
        gp.evolve()

        assert gp.final_population

    def test_ge(self):
        population_size = 11

        grammar = extract_grammar([Option, Rec], Root)
        gp = GP(
            representation=GrammaticalEvolutionRepresentation(grammar=grammar, max_depth=5),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            stopping_criterium=GenerationStoppingCriterium(2),
            population_size=population_size,
            initializer=RampedHalfAndHalfInitializer(),
        )
        gp.evolve()

        assert gp.final_population
