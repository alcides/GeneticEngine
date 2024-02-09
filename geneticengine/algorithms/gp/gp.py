from __future__ import annotations
from geneticengine.algorithms.heuristics import HeuristicSearch


from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.recorder import SingleObjectiveProgressTracker
from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.initializers import (
    StandardInitializer,
)
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import RepresentationWithMutation, SolutionRepresentation


def default_generic_programming_step():
    """The default step in Genetic Programming."""
    return ParallelStep(
        [
            ElitismStep(),
            NoveltyStep(),
            SequenceStep(
                TournamentSelection(5),
                GenericCrossoverStep(0.01),
                GenericMutationStep(0.9),
            ),
        ],
        weights=[5, 5, 90],
    )


class GeneticProgramming(HeuristicSearch):
    """Represents the Genetic Programming algorithm. Defaults as given in A
    Field Guide to GP, p.17, by Poli and Mcphee:

    Args:
        problem (Problem): A SingleObjectiveProblem or a MultiObjectiveProblem problem.
        budget (SearchBudget): how long to search for
        representation (Representation): The individual representation used by the GP program.
        random (RandomSource): A RNG instance
        recorder (SingleObjectiveProgressTracker): How to record the results of evaluations
        population_size (int): The population size (default = 200).
        population_initializer (PopulationInitializer): The method to generate new individuals.
        step (GeneticStep): The main structure of evolution.
    """

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: SolutionRepresentation,
        random: RandomSource = None,
        recorder: SingleObjectiveProgressTracker | None = None,
        population_size: int = 100,
        population_initializer: PopulationInitializer = None,
        step: GeneticStep | None = None,
    ):
        super().__init__(problem, budget, representation, random, recorder)
        self.population_size = population_size
        self.population_initializer = (
            population_initializer if population_initializer is not None else StandardInitializer()
        )
        self.step = step if step is not None else default_generic_programming_step()

    def search(self) -> Individual:
        assert isinstance(self.representation, RepresentationWithMutation)
        # TODO: Crossover
        generation = 0
        population = self.population_initializer.initialize(
            self.problem,
            self.representation,
            self.random,
            self.population_size,
        )
        self.tracker.evaluate(population)
        while not self.is_done():
            generation += 1
            population = self.step.iterate(
                self.problem,
                self.tracker.evaluator,
                self.representation,
                self.random,
                population,
                self.population_size,
                generation,
            )
            self.tracker.evaluate(population)

        return self.tracker.get_best_individual()
