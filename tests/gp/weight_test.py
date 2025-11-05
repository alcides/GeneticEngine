

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.decorators import weight
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.evaluation.recorder import SearchRecorder
from geneticengine.solutions.individual import Individual
from geneticengine.evaluation.tracker import ProgressTracker
from typing import List

@abstract
class Option:
    pass

@weight(0.1)
class OptionA(Option):
    pass

@weight(0.9)
class OptionB(Option):
    pass

class EvaluationRecorder(SearchRecorder):
    def __init__(self):
        self.evaluated_individuals: List[Individual] = []
    def register(self, tracker, individual: Individual, problem, is_best:bool) -> None:
        self.evaluated_individuals.append(individual)
    def get_evaluated_individuals(self) -> List[Individual]:
        return self.evaluated_individuals

class TestWeightedGrammar:
    def test_weighted_grammar_maxdepthdecider(self):
        g = extract_grammar([OptionA, OptionB], Option)
        r = NativeRandomSource(0)
        my_recorder = EvaluationRecorder()
        prob = SingleObjectiveProblem(lambda p: 0)
        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(grammar=g, decider=MaxDepthDecider(r, g, 1)),
            problem=prob,
            population_size=1000,
            budget=EvaluationBudget(50*1000),
            tracker=ProgressTracker(problem=prob, recorders=[my_recorder]),
        )
        gp.search()
        all_evaluated = my_recorder.get_evaluated_individuals()
        countA = sum(1 for ind in all_evaluated if isinstance(ind.get_phenotype(), OptionA))
        countB = sum(1 for ind in all_evaluated if isinstance(ind.get_phenotype(), OptionB))

        weightA = g.get_weights().get(OptionA, 1)
        weightB = g.get_weights().get(OptionB, 1)

        total_evaluations = len(all_evaluated)

        expectedA = total_evaluations * (weightA / (weightA + weightB))
        expectedB = total_evaluations * (weightB / (weightA + weightB))

        tolerance = 0.05

        assert abs(countA - expectedA) / total_evaluations < tolerance
        assert abs(countB - expectedB) / total_evaluations < tolerance
