from functools import reduce
from typing import Iterable, Iterator, Optional, TypeVar
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.combinators import IdentityStep, ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.algorithms.gp.structure import GeneticStep, PopulationInitializer
from geneticengine.evaluation.api import Evaluator
from geneticengine.evaluation.budget import AnyOf, SearchBudget, TimeBudget
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.problems import Fitness, Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.solutions.individual import PhenotypicIndividual, Individual


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


class AjustPopulationSizeStep(IdentityStep):

    last_best: Fitness

    def __init__(self, pgp):
        self.pgp = pgp
        self.first_iteration = True
        self.last_improvement = -1

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
        evaluator.evaluate(problem, population)
        best = self.pgp.tracker.get_best_individuals()[0].get_fitness(problem)
        if self.first_iteration:
            self.first_iteration = False
        elif problem.is_better(best, self.last_best):
            self.last_improvement = generation
        else:
            change_r = random.randint(0, generation)
            if change_r > self.last_improvement:
                self.pgp.population_size = generate_random_population_size(random)
                self.last_improvement = generation

        self.last_best = best


class FeedbackParallelStep(ParallelStep):
    def __init__(
        self,
        tracker: ProgressTracker,
        steps: list[GeneticStep],
        weights: list[float] | None = None,
    ):
        super().__init__(steps, weights)
        self.tracker = tracker

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        npopulation: list[PhenotypicIndividual] = [i for i in population]
        ranges = self.compute_ranges(npopulation, target_size)
        assert len(ranges) == len(self.steps)

        # Compute average fitness
        npopulation = list(evaluator.evaluate(problem, npopulation))
        old_best_fitness = self.tracker.get_best_individuals()[0].get_fitness(problem)

        for (start, end), step, i in zip(ranges, self.steps, range(len(self.steps))):
            if end - start > 0:
                new_individuals = list(
                    step.apply(
                        problem,
                        evaluator,
                        representation,
                        random,
                        population,
                        end - start,
                        generation,
                    ),
                )
                new_individuals = list(evaluator.evaluate(problem, new_individuals))
                delta = sum(
                    1 if problem.is_better(ind.get_fitness(problem), old_best_fitness) else 0 for ind in new_individuals
                )
                self.weights[i] += delta
                yield from new_individuals


T = TypeVar("T", bound=Individual)
def best_of_population(population: Iterable[T], problem: Problem) -> T:
    return reduce(
        lambda x, s: x if problem.is_better(x.get_fitness(problem), s.get_fitness(problem)) else s,
        population,
    )


class GenericAdaptiveMutationStep(GenericMutationStep):

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
        npop = evaluator.evaluate(problem, [i for i in population])
        best = best_of_population(npop, problem)
        best_fitness = best.get_fitness(problem)
        if self.first:
            self.first = False
        elif problem.is_better(best_fitness, self.last_fitness):
            pass
        else:
            self.probability = random.random_float(0.0, 1.0)
        self.last_fitness = best_fitness


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
        npop : Iterable[PhenotypicIndividual] = evaluator.evaluate(problem, [i for i in population])
        best = best_of_population(npop, problem)
        best_fitness = best.get_fitness(problem)
        if self.first:
            self.first = False
        elif problem.is_better(best_fitness, self.last_fitness):
            pass
        else:
            self.probability = random.random_float(0.0, 1.0)
        self.last_fitness = best_fitness


class AdaptiveGeneticProgramming(GeneticProgramming):
    """A Genetic Programming version that automatically adjusts population size, operator probabilities and weights between alternative operators."""

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
        self.mutation = GenericAdaptiveMutationStep(random.random_float(0.0, 1.0))

        self.crossover_tournament = TournamentSelection(random.randint(2, self.population_size))
        self.crossover = GenericAdaptiveCrossoverStep(random.random_float(0.0, 1.0))

        self.step = SequenceStep(
            FeedbackParallelStep(
                self.tracker,
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
                weights=4 * [self.population_size * 1.0],
            ),
            AjustPopulationSizeStep(self),
        )
