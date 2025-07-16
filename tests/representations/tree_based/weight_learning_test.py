from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from geneticengine.grammar import extract_grammar
from geneticengine.prelude import abstract
from geneticengine.grammar.decorators import weight
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.combinators import SequenceStep, ParallelStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.evaluation.budget import TimeBudget, EvaluationBudget
from geneticengine.algorithms.gp.operators.weight_learning import WeightLearningStep, ConditionalWeightLearningStep
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
    good_count = 0
    bad_count = 0
    print(n)
    q: list[Any] = [n]
    while q:
        curr = q.pop(0)
        if isinstance(curr, Good):
            good_count += 1
        elif isinstance(curr, Bad):
            bad_count += 1

        if hasattr(curr, "__dict__"):
            for field in curr.__dict__.values():
                if isinstance(field, Node):
                    q.append(field)
    return float(good_count - bad_count)

def low_fitness_function(n: Node) -> float:
    return 0.1

class TestWeightLearning:
    def test_weight_learning_adapts_grammar(self):
        grammar = extract_grammar([Good, Bad, Leaf], Node)
        problem = SingleObjectiveProblem(fitness_function=fitness_function, minimize=False)
        representation = TreeBasedRepresentation(grammar, decider=MaxDepthDecider(NativeRandomSource(0), grammar, max_depth=5))

        gp_params = {
            "population_size": 10,
            "n_elites": 2,
            "novelty_size": 5,
            "tournament_size": 3,
            "crossover_probability": 0.9,
            "mutation_probability": 0.1,
            "learning_rate": 0.01,
        }

        main_evolution_step = ParallelStep(
            [
                ElitismStep(),
                NoveltyStep(),
                SequenceStep(
                    TournamentSelection(gp_params["tournament_size"]),
                    GenericCrossoverStep(gp_params["crossover_probability"]),
                    GenericMutationStep(gp_params["mutation_probability"]),
                ),
            ],
            weights=[
                gp_params["n_elites"],
                gp_params["novelty_size"],
                gp_params["population_size"] - gp_params["n_elites"] - gp_params["novelty_size"],
            ],
        )

        gp_step = SequenceStep(
            WeightLearningStep(gp_params["learning_rate"]),
            main_evolution_step,
        )

        alg = GeneticProgramming(
            problem=problem,
            representation=representation,
            step=gp_step,
            population_size=gp_params["population_size"],
            budget=EvaluationBudget(100),
            random=NativeRandomSource(0),
        )

        all_initial_weights = grammar.get_weights()
        initial_weights = {prod_class: all_initial_weights[prod_class] for prod_class in grammar.alternatives[Node]}
        print(initial_weights)

        alg.search()[0]

        all_final_weights = grammar.get_weights()
        final_weights = {prod_class: all_final_weights[prod_class] for prod_class in grammar.alternatives[Node]}

        assert final_weights[Good] > initial_weights[Good], "Good weights should increase"
        assert final_weights[Bad] < initial_weights[Bad], "Bad weights should decrease"

    def test_conditional_weight_learning_does_not_trigger(self):
        """Tests that ConditionalWeightLearningStep does nothing if fitness is below threshold."""
        grammar = extract_grammar([Good, Bad, Leaf], Node)
        # Use the fitness function that always returns a low score
        problem = SingleObjectiveProblem(
            fitness_function=low_fitness_function, minimize=False,
        )
        representation = TreeBasedRepresentation(
            grammar, decider=MaxDepthDecider(NativeRandomSource(0), grammar, max_depth=5),
        )

        gp_step = ConditionalWeightLearningStep(fitness_threshold=0.5)

        alg = GeneticProgramming(
            problem=problem,
            representation=representation,
            step=gp_step,
            population_size=5,
            budget=TimeBudget(5),
            random=NativeRandomSource(0),
        )

        all_initial_weights = grammar.get_weights()
        initial_weights = {
            prod_class: all_initial_weights[prod_class]
            for prod_class in grammar.alternatives[Node]
        }

        alg.search()

        all_final_weights = grammar.get_weights()
        final_weights = {
            prod_class: all_final_weights[prod_class]
            for prod_class in grammar.alternatives[Node]
        }

        # Assert that the weights have not changed
        assert final_weights == initial_weights

    def test_conditional_weight_learning_triggers(self):
        """Tests that ConditionalWeightLearningStep works if fitness is above threshold."""
        grammar = extract_grammar([Good, Bad, Leaf], Node)
        # Use the original fitness function that can exceed the threshold
        problem = SingleObjectiveProblem(fitness_function=fitness_function, minimize=False)
        representation = TreeBasedRepresentation(
            grammar, decider=MaxDepthDecider(NativeRandomSource(0), grammar, max_depth=5),
        )

        gp_step = ConditionalWeightLearningStep(fitness_threshold=0.5)

        alg = GeneticProgramming(
            problem=problem,
            representation=representation,
            step=gp_step,
            population_size=5,
            budget=TimeBudget(5),
            random=NativeRandomSource(0),
        )

        all_initial_weights = grammar.get_weights()
        initial_weights = {
            prod_class: all_initial_weights[prod_class]
            for prod_class in grammar.alternatives[Node]
        }

        alg.search()

        all_final_weights = grammar.get_weights()
        final_weights = {
            prod_class: all_final_weights[prod_class]
            for prod_class in grammar.alternatives[Node]
        }

        # Assert that the weights have changed
        assert final_weights != initial_weights
        assert final_weights[Good] > initial_weights[Good]
