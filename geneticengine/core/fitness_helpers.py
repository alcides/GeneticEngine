from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.problems import Problem


def best_individual(population: list[Individual], problem: Problem) -> Individual:
    """Returns the best individual of a population."""
    return max(population, key=lambda x: -problem.key_function(x))


def is_better(problem: Problem, individual: Individual, other: Individual) -> bool:
    """Returns whether one individual is better than other."""
    return problem.is_better(individual.get_fitness(problem), other.get_fitness(problem))


def sort_population(population: list[Individual], problem: Problem) -> list[Individual]:
    return sorted(population, key=problem.fitness_function, reverse=True)
