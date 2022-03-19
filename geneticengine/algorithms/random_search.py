from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Type

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import relabel_nodes_of_trees
from geneticengine.core.representations.tree.treebased import treebased_representation


class RandomSearch:
    """
    Random Search object. Main attribute: evolve

    Parameters:
        - grammar (Grammar): The grammar used to guide the search.
        - evaluation_function (Callable[[Any], float]): The fitness function. Should take in any valid individual and return a float. The default is that the higher the fitness, the more applicable is the solution to the problem. Turn on the parameter minimize to switch it around.
        - minimize (bool): When switch on, the fitness function is reversed, so that a higher result from the fitness function corresponds to a less fit solution (default = False).
        - representation (Representation): The individual representation used by the GP program. The default is treebased_representation.
        - randomSource (Callable[[int], RandomSource]): The random source function used by the program. Should take in an integer, representing the seed, and return a RandomSource.
        - seed (int): The seed of the RandomSource (default = 123).
        - population_size (int): The population size (default = 200). Apart from the first generation, each generation the population is made up of the elites, novelties, and transformed individuals from the previous generation. Note that population_size > (n_elites + n_novelties + 1) must hold.
        - number_of_generations (int): Number of generations (default = 100).
        - max_depth (int): The maximum depth a tree can have (default = 15).
        - favor_less_deep_trees (bool): If set to True, this gives a tiny penalty to deeper trees to favor simpler trees (default = False).
        - force_individual (Any): Allows the incorporation of an individual in the first population (default = None).

    """

    def __init__(
        self,
        grammar: Grammar,
        evaluation_function: Callable[[Any], float],
        representation: Representation = treebased_representation,
        randomSource: Callable[[int], RandomSource] = RandomSource,
        population_size: int = 200,
        number_of_generations: int = 100,
        max_depth: int = 15,
        # now based on depth, maybe on number of nodes?
        favor_less_deep_trees: bool = False,
        minimize: bool = False,
        force_individual: Any = None,
        seed: int = 123,
    ):
        assert population_size >= 1

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
            r=self.random,
            g=self.grammar,
            depth=depth,
        )
        return Individual(
            genotype=genotype,
            fitness=None,
        )

    def evaluate(self, individual: Individual) -> float:
        if individual.fitness is None:
            phenotype = self.representation.genotype_to_phenotype(
                self.grammar,
                individual.genotype,
            )
            individual.fitness = self.evaluation_function(phenotype)
        return individual.fitness

    def keyfitness(self):
        if self.minimize:
            return lambda x: self.evaluate(x)
        else:
            return lambda x: -self.evaluate(x)

    def evolve(self, verbose=1):
        best = None
        best_ind = None
        if self.force_individual is not None:
            best_ind = Individual(
                genotype=relabel_nodes_of_trees(
                    self.force_individual,
                    self.grammar,
                ),
                fitness=None,
            )
            best = self.keyfitness()(best_ind)
        for gen in range(self.number_of_generations):
            for _ in range(self.population_size):
                i = self.create_individual(15)
                f = self.keyfitness()(i)
                if (best == None) or (f < best):
                    best = f
                    best_ind = i
            if verbose == 2:
                print(f"Best population:{best_ind}.")
            if verbose >= 1:
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
                self.grammar,
                best_ind.genotype,
            ),
        )
