from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.solutions.individual import Individual
from geneticengine.evaluation import Evaluator
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.random.sources import NativeRandomSource, RandomSource

from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.problems import Problem, SingleObjectiveProblem

from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.algorithms.gp.gp import GeneticProgramming

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.metahandlers.lists import ListSizeBetween


class NonTerminal(ABC):
    pass


@dataclass
class Root:
    options: Annotated[list[NonTerminal], ListSizeBetween(1, 3)]


@dataclass
class OptionA(NonTerminal):
    value: int


@dataclass
class OptionB(NonTerminal):
    value: float


class CustomMutationOperator(MutationOperator[Root]):
    def mutate(
        self,
        genotype: Root,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: RandomSource,
        index_in_population: int,
        generation: int,
    ) -> Root:
        cpy = deepcopy(genotype)
        el = random_source.randint(0, len(cpy.options) - 1)
        to_be_mutated = cpy.options[el]
        if isinstance(to_be_mutated, OptionA):
            to_be_mutated.value += 1
        elif isinstance(to_be_mutated, OptionB):
            to_be_mutated.value += 0.1
        else:
            assert False
        return cpy


def algorithm_steps():
    """The default step in Genetic Programming."""
    return ParallelStep(
        [
            ElitismStep(),
            SequenceStep(
                TournamentSelection(10),
                GenericCrossoverStep(1),
                GenericMutationStep(1, operator=CustomMutationOperator()),
            ),
        ],
        weights=[1, 99],
    )


def test_custom_mutation_baseline():
    g = extract_grammar([OptionA, OptionB], Root)
    p = SingleObjectiveProblem(lambda x: 1)
    alg = GeneticProgramming(
        representation=TreeBasedRepresentation(g, 2),
        problem=p,
        budget=EvaluationBudget(2000),
        step=algorithm_steps(),
        population_size=200,
    )
    ind = alg.search()
    assert ind


def test_custom_mutation():
    g = extract_grammar([OptionA, OptionB], Root)
    repr = TreeBasedRepresentation(g, 2)
    p = SingleObjectiveProblem(lambda x: 1)
    population = [
        Individual(
            genotype=Root([OptionA(value=1), OptionB(value=2.0)]),
            genotype_to_phenotype=repr.genotype_to_phenotype,
        ),
    ]

    custom_mutation_step = GenericMutationStep(1, operator=CustomMutationOperator())
    new_population = custom_mutation_step.iterate(
        p,
        SequentialEvaluator(),
        repr,
        NativeRandomSource(3),
        population,
        10,
        0,
    )

    for ind in new_population:
        ph = ind.get_phenotype()
        print(ph.options)
        assert isinstance(ph, Root)
        assert len(ph.options) == 2
        assert isinstance(ph.options[0], OptionA)
        assert isinstance(ph.options[1], OptionB)
        assert (ph.options[0].value == 1 and ph.options[1].value != 2.0) or (
            ph.options[0].value != 1 and ph.options[1].value == 2.0
        )
        assert -4 <= ph.options[0].value <= 6
        assert 1 <= ph.options[1].value <= 3
