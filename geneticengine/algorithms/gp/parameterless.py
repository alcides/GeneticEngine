from functools import reduce
from typing import Iterator, Optional, TypeVar
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.combinators import IdentityStep, ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.evaluation.api import Evaluator
from geneticengine.evaluation.budget import AnyOf, SearchBudget, TimeBudget
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.problems import Fitness, Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.solutions.individual import Individual, PhenotypicIndividual


def generate_random_population_size(random: RandomSource) -> int:
    return random.randint(2, 1000)


def time_for_initialization(budget: SearchBudget) -> Optional[float]:
    match budget:
        case TimeBudget():
            return budget.time_budget * 0.001
        case AnyOf():
            v1 = time_for_initialization(budget.a)
            if v1 is not None:
                return v1
            return time_for_initialization(budget.b)
        case _:
            return None


class ParameterlessPopulationInitializer(PopulationInitializer):
    def __init__(self, budget: SearchBudget, tracker: ProgressTracker):

        self.budget = TimeBudget(time_for_initialization(budget) or 1)
        self.tracker = tracker

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random: RandomSource,
        target_size: int,
        **kwargs,
    ) -> Iterator[PhenotypicIndividual]:
        while not self.budget.is_done(self.tracker):
            yield PhenotypicIndividual(
                representation.create_genotype(
                    random,
                    **kwargs,
                ),
                representation=representation,
            )


class RandomizeParallelStep(ParallelStep):

    def post_iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> None:
        self.weights = [random.randint(0, 1000) for _ in range(4)]


T = TypeVar("T", bound=Individual)
def best_of_population(population: Iterator[T], problem: Problem) -> Individual:
    return reduce(
        lambda x, s: x if problem.is_better(x.get_fitness(problem), s.get_fitness(problem)) else s,
        list(population),
    )


class RegenerateWeightsStep(IdentityStep):

    def __init__(
        self,
        mut: GenericMutationStep,
        xo: GenericCrossoverStep,
        tmut: TournamentSelection,
        txo: TournamentSelection,
    ):
        self.mut = mut
        self.xo = xo
        self.tmut = tmut
        self.txo = txo

    def post_iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> None:
        self.mut.probability = random.random_float(0.0, 1.0)
        self.xo.probability = random.random_float(0.0, 1.0)
        self.tmut.tournament_size = random.randint(2, target_size)
        self.txo.tournament_size = random.randint(2, target_size)


class GenericAdaptiveCrossoverStep(GenericCrossoverStep):

    last_fitness: Fitness

    def __init__(self, probability: float = 1):
        super().__init__(probability)
        self.first = True

    def post_iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> None:
        candidates = list(evaluator.evaluate(problem, population))
        best = best_of_population(iter(candidates), problem)
        best_fitness = best.get_fitness(problem)
        if self.first:
            self.first = False
        elif problem.is_better(best_fitness, self.last_fitness):
            pass
        else:
            self.probability = random.random_float(0.0, 1.0)
        self.last_fitness = best_fitness


class InitiallyRandomGeneticProgramming(GeneticProgramming):
    """A Genetic Programming version that uses random configurations, set before the evolution."""

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: Representation,
        random: RandomSource = None,
        tracker: ProgressTracker | None = None,
    ):
        super().__init__(problem, budget, representation, random, tracker)
        self.population_initializer = ParameterlessPopulationInitializer(self.budget, self.tracker)
        self.population_size = generate_random_population_size(self.random)
        self.mutation_tournament = TournamentSelection(random.randint(2, self.population_size))
        self.mutation = GenericMutationStep(random.random_float(0.0, 1.0))

        self.crossover_tournament = TournamentSelection(random.randint(2, self.population_size))
        self.crossover = GenericCrossoverStep(random.random_float(0.0, 1.0))

        self.step = ParallelStep(
            [
                ElitismStep(),
                NoveltyStep(),
                SequenceStep(
                    self.mutation_tournament,
                    self.mutation,
                ),
                SequenceStep(
                    self.crossover_tournament,
                    self.crossover,
                ),
            ],
            weights=[random.randint(0, 1000) for _ in range(4)],
        )


class AlwaysRandomGeneticProgramming(GeneticProgramming):
    """A Genetic Programming version that uses random configurations, which are regenerated every generation."""

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: Representation,
        random: RandomSource = None,
        tracker: ProgressTracker | None = None,
    ):
        super().__init__(problem, budget, representation, random, tracker)
        self.population_initializer = ParameterlessPopulationInitializer(self.budget, self.tracker)
        self.population_size = generate_random_population_size(self.random)
        self.mutation_tournament = TournamentSelection(random.randint(2, self.population_size))
        self.mutation = GenericMutationStep(random.random_float(0.0, 1.0))

        self.crossover_tournament = TournamentSelection(random.randint(2, self.population_size))
        self.crossover = GenericCrossoverStep(random.random_float(0.0, 1.0))

        self.step = SequenceStep(
            RandomizeParallelStep(
                [
                    ElitismStep(),
                    NoveltyStep(),
                    SequenceStep(
                        self.mutation_tournament,
                        self.mutation,
                    ),
                    SequenceStep(
                        self.crossover_tournament,
                        self.crossover,
                    ),
                ],
                weights=[random.randint(0, 1000) for _ in range(4)],
            ),
            RegenerateWeightsStep(self.mutation, self.crossover, self.mutation_tournament, self.crossover_tournament),
        )
