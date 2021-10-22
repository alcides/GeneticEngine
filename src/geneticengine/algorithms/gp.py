from dataclasses import dataclass
from typing import Any, Callable, Generic, List, Optional, Protocol, Tuple, TypeVar
from copy import deepcopy
from geneticengine.core.tree import Node
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.base import Representation
from geneticengine.core.representations.treebased import treebased_representation


@dataclass
class Individual(object):
    genotype: Any
    fitness: Optional[float] = None

    def __str__(self) -> str:
        return str(self.genotype)


def create_tournament(
    evaluation_function: Callable[[Node], float], tournament_size: int, minimize=False
) -> Callable[[RandomSource, List[Individual], int], List[Individual]]:
    def tournament(
        r: RandomSource, population: List[Individual], n: int
    ) -> List[Individual]:
        winners = []
        for _ in range(n):
            candidates = [r.choice(population) for _ in range(tournament_size)]
            winner = candidates[0]
            for o in candidates[1:]:
                if o.fitness > winner.fitness and not minimize:
                    winner = o
                if o.fitness < winner.fitness and minimize:
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
        novelty: int = 10,
        number_of_generations: int = 100,
        max_depth: int = 15,
        selection: Tuple[str, int] = ("tournament", 5),
        # -----
        # As given in A Field Guide to GP, p.17, by Poli and Mcphee
        probability_mutation: float = 0.01,
        probability_crossover: float = 0.9,
        # -----
        minimize: bool = False,
        force_individual: Any = None
    ):
        self.grammar = representation.preprocess_grammar(g)
        self.representation = representation
        self.evaluation_function = evaluation_function
        self.random = RandomSource(123)
        self.population_size = population_size
        self.elitism = elitism
        self.novelty = novelty
        self.number_of_generations = number_of_generations
        self.max_depth = max_depth
        self.probability_mutation = probability_mutation
        self.probability_crossover = probability_crossover
        self.minimize = minimize
        if selection[0] == "tournament":
            self.selection = create_tournament(
                self.evaluation_function, selection[1], self.minimize
            )
        else:
            self.selection = lambda r, ls, n: [x for x in ls[:n]]
        self.force_individual = force_individual

    def create_individual(self):
        return Individual(
            genotype=self.representation.create_individual(
                self.random, self.grammar, self.max_depth
            ),
            fitness=None,
        )

    def selectOne(self, population):
        return self.selection(self.random, population, 1)[0]

    def evaluate(self, individual: Individual) -> float:
        if individual.fitness is None:
            individual.fitness = self.evaluation_function(individual.genotype)
        return individual.fitness

    def evolve(self):
        if self.minimize:
            keyfitness = lambda x: self.evaluate(x)
        else:
            keyfitness = lambda x: -self.evaluate(x)

        population = [self.create_individual() for _ in range(self.population_size)]
        if self.force_individual is not None:
            population[0] = Individual(
            genotype=self.force_individual,
            fitness=None,
        )
        population = sorted(population, key=keyfitness)

        for gen in range(self.number_of_generations):
            npop = [self.create_individual() for _ in range(self.novelty)]
            npop.extend([deepcopy(x) for x in population[: self.elitism]])
            spotsLeft = self.population_size - self.elitism - self.novelty
            for _ in range(spotsLeft // 2):
                p1 = self.selectOne(population)
                p2 = self.selectOne(population)
                if self.random.randint(0, 100) < self.probability_crossover * 100:
                    # Crossover
                    (g1, g2) = self.representation.crossover_individuals(
                        self.random, self.grammar, p1.genotype, p2.genotype
                    )
                    p1 = Individual(g1)
                    p2 = Individual(g2)
                if self.random.randint(0, 100) < self.probability_mutation * 100:
                    p1 = Individual(
                        genotype=self.representation.mutate_individual(
                            self.random, self.grammar, p1.genotype
                        ),
                        fitness=None,
                    )
                if self.random.randint(0, 100) < self.probability_mutation * 100:
                    p2 = Individual(
                        genotype=self.representation.mutate_individual(
                            self.random, self.grammar, p2.genotype
                        ),
                        fitness=None,
                    )
                npop.append(p1)
                npop.append(p2)

            population = npop
            population = sorted(population, key=keyfitness)
            # self.printFitnesses(population, "G:" + str(gen))
            print(
                "BEST at",
                gen,
                "/",
                self.number_of_generations,
                "is",
                round(self.evaluate(population[0]), 2),
                # population[0]
            )
        return (population[0], self.evaluate(population[0]))

    def printFitnesses(self, pop, prefix):
        print(prefix)
        for x in pop:
            print(round(self.evaluate(x), 0), str(x))
        print("---")
