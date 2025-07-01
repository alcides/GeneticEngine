from typing import Iterator, TypeVar
from geneticengine.solutions.individual import Individual
from geneticengine.problems import Problem

T = TypeVar("T", bound=Individual)

def dominates(ind:T, other:T, problem:Problem):
    this_scores = ind.get_fitness(problem).fitness_components
    other_scores = other.get_fitness(problem).fitness_components
    return all(i <= j if m else i >= j for i, j, m in zip(this_scores, other_scores, problem.minimize)) and any(i < j if m else i > j for i, j, m in zip(this_scores, other_scores, problem.minimize))

def non_dominated(population: Iterator[T], problem: Problem) -> Iterator[T]:
    """Returns the best individual of a population."""
    pop = list(population)
    yield from (
        ind
        for ind in pop
        if not any( dominates(other_ind, ind, problem) for other_ind in pop)
    )


def is_better(problem: Problem, individual: Individual, other: Individual) -> bool:
    """Returns whether one individual is better than other.

    Requires the individuals to be evaluated.
    """
    return problem.is_better(individual.get_fitness(problem), other.get_fitness(problem))
