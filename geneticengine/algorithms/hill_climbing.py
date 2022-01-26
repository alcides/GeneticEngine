from typing import Any, Callable, Type
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.algorithms.gp.Individual import Individual
import geneticengine.algorithms.gp.generation_steps.mutation as mutation


class HC(object):
    def __init__(
        self,
        g: Grammar,
        evaluation_function: Callable[[Any], float],
        representation: Representation = treebased_representation,
        random_source_type: Type = RandomSource,
        population_size: int = 200,
        number_of_generations: int = 100,
        max_depth: int = 15,
        minimize: bool = False,
        force_individual: Any = None,
        seed: int = 123,
    ):
        # Add check to input numbers (n_elitism, n_novelties, population_size)
        self.grammar: Grammar = g
        self.representation = representation
        self.evaluation_function = evaluation_function
        self.random = random_source_type(seed)
        self.population_size = population_size
        self.minimize = minimize
        self.mutation = mutation.create_hill_climbing_mutation(
            self.random,
            self.representation,
            self.grammar,
            max_depth,
            self.keyfitness(),
            population_size,
        )
        self.number_of_generations = number_of_generations
        if force_individual is not None:
            self.population = Individual(
                genotype=force_individual,
                fitness=None,
            )
        else:
            self.population = Individual(
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
        population = self.population

        for gen in range(self.number_of_generations):

            population = self.mutation(population)
            print(
                "BEST at",
                gen + 1,
                "/",
                self.number_of_generations,
                "is",
                round(self.evaluate(population), 2),
            )
        return (
            population,
            self.evaluate(population),
            self.representation.genotype_to_phenotype(
                self.grammar, population.genotype
            ),
        )
