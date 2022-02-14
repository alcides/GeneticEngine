from typing import Callable, Any, Type
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import (
    treebased_representation,
    relabel_nodes_of_trees,
)
from geneticengine.algorithms.gp.individual import Individual


class RandomSearch(object):
    def __init__(
        self,
        grammar: Grammar,
        evaluation_function: Callable[[Any], float],
        representation: Representation = treebased_representation,
        randomSource: Callable[[int], RandomSource] = RandomSource,
        population_size: int = 200,
        number_of_generations: int = 100,
        max_depth: int = 15,
        favor_less_deep_trees: bool = False,  # now based on depth, maybe on number of nodes?
        minimize: bool = False,
        force_individual: Any = None,
        seed: int = 123,
    ):
        # Add check to input numbers (n_elitism, n_novelties, population_size)
        self.grammar = grammar
        self.representation = representation
        self.evaluation_function = evaluation_function
        self.random = randomSource(seed)
        self.seed = seed
        self.population_size = population_size
        self.max_depth = max_depth
        self.favor_less_deep_trees = favor_less_deep_trees
        self.minimize = minimize
        self.number_of_generations = number_of_generations
        self.force_individual = force_individual

    def create_individual(self, depth: int):
        genotype = self.representation.create_individual(
            r=self.random, g=self.grammar, depth=depth
        )
        return Individual(
            genotype=genotype,
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
        best = 1000000
        best_ind = None
        if self.force_individual is not None:
            best_ind = Individual(
                genotype=relabel_nodes_of_trees(
                    self.force_individual, self.grammar.non_terminals, self.max_depth
                ),
                fitness=None,
            )
            best = self.keyfitness()(best_ind)
        for gen in range(self.number_of_generations):
            for _ in range(self.population_size):
                i = self.create_individual(15)
                f = self.keyfitness()(i)
                if f < best:
                    best = f
                    best_ind = i
            if verbose == 1:
                print("Best population:{}.".format(best_ind))
            print(
                "BEST at",
                gen + 1,
                "/",
                self.number_of_generations,
                "is",
                round(best, 2),
            )

        return (
            best_ind,
            self.evaluate(best_ind),
            self.representation.genotype_to_phenotype(
                self.grammar, best_ind.genotype
            ),
        )
