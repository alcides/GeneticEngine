from typing import Any, Callable, Generic, List, Optional, Protocol, Tuple, TypeVar
from copy import deepcopy
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.base import Representation
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.algorithms.gp.Individual import Individual
import geneticengine.algorithms.gp.generation_steps.selection as selection
import geneticengine.algorithms.gp.generation_steps.mutation as mutation
import geneticengine.algorithms.gp.generation_steps.cross_over as cross_over


class GP(object):
    def __init__(
        self,
        g: Grammar,
        representation: Representation,
        evaluation_function: Callable[[Any], float],
        randomSource: RandomSource = RandomSource,
        population_size: int = 200,
        n_elites: int = 5,  # Shouldn't this be a percentage of population size?
        n_novelties: int = 10,
        number_of_generations: int = 100,
        max_depth: int = 15,
        selection_method: Tuple[str, int] = ("tournament", 5),
        # -----
        # As given in A Field Guide to GP, p.17, by Poli and Mcphee
        probability_mutation: float = 0.01,
        probability_crossover: float = 0.9,
        # -----
        minimize: bool = False,
        force_individual: Any = None,
        seed: int = 123,
    ):
        # Add check to input numbers (n_elitism, n_novelties, population_size)
        self.grammar: Grammar = g
        self.representation = representation
        self.evaluation_function = evaluation_function
        self.random = randomSource(seed)
        self.population_size = population_size
        self.elitism = selection.create_elitism(n_elites)
        self.novelty = selection.create_novelties(
            self.create_individual, max_depth=max_depth
        )
        self.mutation = mutation.create_mutation(
            self.random, self.representation, self.grammar, max_depth
        )
        self.cross_over = cross_over.create_cross_over(
            self.random, self.representation, self.grammar, max_depth
        )
        self.n_novelties = n_novelties
        self.number_of_generations = number_of_generations
        self.probability_mutation = probability_mutation
        self.probability_crossover = probability_crossover
        self.minimize = minimize
        if selection_method[0] == "tournament":
            self.selection = selection.create_tournament(
                selection_method[1], self.minimize
            )
        else:
            self.selection = lambda r, ls, n: [x for x in ls[:n]]
        self.force_individual = force_individual

    def create_individual(self, max_depth):
        return Individual(
            genotype=self.representation.create_individual(
                self.random, self.grammar, max_depth
            ),
            fitness=None,
        )

    def evaluate(self, individual: Individual) -> float:
        if individual.fitness is None:
            phenotype = self.representation.genotype_to_phenotype(
                self.grammar, individual.genotype
            )
            individual.fitness = self.evaluation_function(phenotype)
        return individual.fitness

    def keyfitness(self):
        if self.minimize:
            return lambda x: self.evaluate(x)
        else:
            return lambda x: -self.evaluate(x)

    def evolve(self, verbose=0):
        # TODO: This is not ramped half and half
        population = [
            self.create_individual(self.grammar.get_max_node_depth())
            for _ in range(self.population_size)
        ]
        if self.force_individual is not None:
            population[0] = Individual(
                genotype=self.force_individual,
                fitness=None,
            )
        population = sorted(population, key=self.keyfitness())

        for gen in range(self.number_of_generations):
            npop = self.novelty(self.n_novelties)
            npop.extend(self.elitism(population, self.keyfitness()))
            spotsLeft = self.population_size - len(npop)
            for _ in range(spotsLeft // 2):
                candidates = self.selection(self.random, population, 2)
                (p1, p2) = candidates[0], candidates[1]
                if self.random.randint(0, 100) < self.probability_crossover * 100:
                    (p1, p2) = self.cross_over(p1, p2)
                if self.random.randint(0, 100) < self.probability_mutation * 100:
                    p1 = self.mutation(p1)
                if self.random.randint(0, 100) < self.probability_mutation * 100:
                    p2 = self.mutation(p2)
                npop.append(p1)
                npop.append(p2)

            population = npop
            population = sorted(population, key=self.keyfitness())
            if verbose == 1:
                self.printFitnesses(population, "G:" + str(gen))
                print("Best population:{}.".format(population[0]))
            print(
                "BEST at",
                gen + 1,
                "/",
                self.number_of_generations,
                "is",
                round(self.evaluate(population[0]), 2),
            )
        return (population[0], self.evaluate(population[0]), self.representation.genotype_to_phenotype(self.grammar,population[0].genotype))

    def printFitnesses(self, pop, prefix):
        print(prefix)
        for x in pop:
            print(round(self.evaluate(x), 0), str(x))
        print("---")
