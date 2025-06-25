from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from geneticengine.grammar import extract_grammar
from geneticengine.prelude import abstract
from geneticengine.grammar.decorators import weight
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.combinators import SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.algorithms.gp.operators.weight_learning import WeightLearningStep
from geneticengine.representations.tree.initializations import MaxDepthDecider

@abstract
class Node:
    pass

@weight(1.0)
@dataclass
class Good(Node):
    child: Node

@weight(1.0)
@dataclass
class Bad(Node):
    child: Node

@weight(1.0)
@dataclass
class Leaf(Node):
    pass

def fitness_function(n: Node) -> float:
    """
    Fitness is the number of "Good" nodes in the tree.
    """
    count = 0

    def count_good_nodes_recursive(node: Any):
        nonlocal count
        if not isinstance(node, Node):
            return

        if isinstance(node, Good):
            count += 1

        if hasattr(node, "__dict__"):
            for field, value in node.__dict__.items():
                if isinstance(value, Node):
                    count_good_nodes_recursive(value)

    count_good_nodes_recursive(n)
    return float(count)

class TestWeightLearning:
    def test_weight_learning_adapts_grammar(self):
        grammar = extract_grammar([Good, Bad, Leaf], Node)
        problem = SingleObjectiveProblem(fitness_function=fitness_function, minimize=False)
        representation = TreeBasedRepresentation(grammar, decider=MaxDepthDecider(NativeRandomSource(42), grammar, max_depth=5))

        gp_step = SequenceStep(
        WeightLearningStep(learning_rate=0.2),
            SequenceStep(
                TournamentSelection(2),
                GenericCrossoverStep(0.8),
                GenericMutationStep(0.2),
            ),
        )

        alg = GeneticProgramming(
            problem=problem,
            representation=representation,
            step=gp_step,
            population_size=50,
            budget=EvaluationBudget(500),
            random=NativeRandomSource(42),
        )

        all_initial_weights = grammar.get_weights()
        initial_weights = {prod_class: all_initial_weights[prod_class] for prod_class in grammar.alternatives[Node]}

        alg.search()

        all_final_weights = grammar.get_weights()
        final_weights = {prod_class: all_final_weights[prod_class] for prod_class in grammar.alternatives[Node]}

        assert final_weights[Good] > initial_weights[Good]
        assert final_weights[Bad] < initial_weights[Bad]
