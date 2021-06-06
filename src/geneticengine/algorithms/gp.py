from typing import Callable, List, Protocol, Tuple, TypeVar
from copy import deepcopy
from geneticengine.core.tree import Node
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.base import Representation
from geneticengine.core.representations.treebased import treebased_representation


class Individual(object):
    def __init__(self, genotype, evaluation_function, representation):
        self.genotype = genotype
        self.fitness = None
        self.evaluation_function = evaluation_function
        self.representation = representation

    def get_fitness(self):
        if self.fitness is None:
            self.fitness = self.evaluation_function(self.genotype)
        return self.fitness

    def mutate(self, random: RandomSource, grammar: Grammar):
        self.genotype = self.representation.mutate_individual(
            random, grammar, self.genotype
        )
        self.fitness = None
        return self


def create_tournament(
    tournament_size: int,
) -> Callable[[RandomSource, List[Individual], int], List[Individual]]:
    def tournament(
        r: RandomSource, population: List[Individual], n: int
    ) -> List[Individual]:
        winners = []
        for _ in range(n):
            candidates = [r.choice(population) for _ in range(tournament_size)]
            winner = candidates[0]
            for o in candidates[1:]:
                if o.get_fitness() > winner.get_fitness():
                    winner = o
            winners.append(winner)
        return winners

    return tournament


class GP(object):
    def __init__(
        self,
        g: Grammar,
        representation: Representation,
        evaluation_function: Callable[[Node], float],
        population_size: int = 200,
        elitism: int = 5,
        novelty: int = 5,
        number_of_generations: int = 1000,
        max_depth: int = 10,
        selection: Callable[
            [RandomSource, List[Individual], int], List[Individual]
        ] = lambda r, pop, n: pop[:n],
        probability_mutation=0.2,
        probability_crossover=0.2,
    ):
        self.grammar = g
        self.representation = representation
        self.evaluation = evaluation_function
        self.random = RandomSource(123)
        self.population_size = population_size
        self.elitism = elitism
        self.novelty = novelty
        self.number_of_generations = number_of_generations
        self.max_depth = max_depth
        self.selection = selection
        self.probability_mutation = probability_mutation
        self.probability_crossover = probability_crossover

    def create_individual(self):
        return Individual(
            self.representation.create_individual(
                self.random, self.grammar, self.max_depth
            ),
            self.evaluation,
            self.representation,
        )

    def selectOne(self, population):
        return self.selection(self.random, population, 1)[0]

    def evolve(self):
        population = [self.create_individual() for _ in range(self.population_size)]
        population = sorted(population, key=lambda x: x.get_fitness())

        for gen in range(self.number_of_generations):
            npop = [self.create_individual() for _ in range(self.novelty)]
            npop.extend([deepcopy(x) for x in population[: self.elitism]])
            spotsLeft = self.population_size - self.elitism - self.novelty
            for _ in range(spotsLeft // 2):
                if self.random.randint(0, 100) < self.probability_crossover * 100:
                    # Crossover
                    (p1, p2) = self.representation.crossover_individuals(
                        self.random, self.grammar, p1, p2
                    )
                else:
                    p1 = self.selectOne(population)
                    p2 = self.selectOne(population)
                if self.random.randint(0, 100) < self.probability_mutation * 100:
                    p1 = p1.mutate(self.random, self.grammar)
                if self.random.randint(0, 100) < self.probability_mutation * 100:
                    p2 = p2.mutate(self.random, self.grammar)
                npop.append(p1)
                npop.append(p2)

            population = npop
            population = sorted(population, key=lambda x: -x.get_fitness())
            print("BEST at", gen, "is", round(population[0].get_fitness(), 0))
        return (population[0], population[0].get_fitness())

    def printFitnesses(self, pop, prefix):
        print(prefix, [round(x.get_fitness(), 0) for x in pop])
