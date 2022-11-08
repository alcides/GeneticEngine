from __future__ import annotations

import time
from typing import Any
from typing import Callable
from typing import Type

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.callbacks.callback import DebugCallback
from geneticengine.algorithms.callbacks.callback import PrintBestCallback
from geneticengine.algorithms.callbacks.callback import ProgressCallback
from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.heuristics import Heuristics
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import FitnessType
from geneticengine.core.problems import Problem
from geneticengine.core.problems import process_problem
from geneticengine.core.problems import wrap_depth
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import relabel_nodes_of_trees
from geneticengine.core.representations.tree.treebased import treebased_representation


class RandomSearch(Heuristics):
    """
    Random Search object. Main attribute: evolve

    Args:
        grammar (Grammar): The grammar used to guide the search.
        evaluation_function (Callable[[Any], float]): The fitness function. Should take in any valid individual and return a float. The default is that the higher the fitness, the more applicable is the solution to the problem. Turn on the parameter minimize to switch it around.
        minimize (bool): When switch on, the fitness function is reversed, so that a higher result from the fitness function corresponds to a less fit solution (default = False).
        representation (Representation): The individual representation used by the GP program. The default is treebased_representation.
        randomSource (Callable[[int], RandomSource]): The random source function used by the program. Should take in an integer, representing the seed, and return a RandomSource.
        seed (int): The seed of the RandomSource (default = 123).
        population_size (int): The population size (default = 200). Apart from the first generation, each generation the population is made up of the elites, novelties, and transformed individuals from the previous generation. Note that population_size > (n_elites + n_novelties + 1) must hold.
        number_of_generations (int): Number of generations (default = 100).
        max_depth (int): The maximum depth a tree can have (default = 15).
        favor_less_complex_trees (bool): If set to True, this gives a tiny penalty to more complex (with more nodes) trees to favor simpler trees (default = False).
        force_individual (Any): Allows the incorporation of an individual in the first population (default = None).
        save_to_csv (str): Saves a CSV file with the details of all the individuals of all generations.
        save_genotype_as_string (bool): Turn this off if you don't want to safe all the genotypes as strings. This saves memory and a bit of time.
        callbacks (List[Callback]): The callbacks to define what is done with the returned prints from the algorithm (default = []).
    """

    def __init__(
        self,
        grammar: Grammar,
        evaluation_function: Callable[[Any], float] | None = None,
        representation: Representation = treebased_representation,
        problem: Problem | None = None,
        randomSource: Callable[[int], RandomSource] = RandomSource,
        population_size: int = 200,
        number_of_generations: int = 100,
        max_depth: int = 15,
        # now based on depth, maybe on number of nodes?
        favor_less_complex_trees: bool = False,
        minimize: bool = False,
        force_individual: Any = None,
        seed: int = 123,
        save_to_csv: str | None = None,
        save_genotype_as_string: bool = True,
        callbacks: list[Callback] | None = None,
    ):
        assert population_size >= 1

        self.problem: Problem = wrap_depth(
            process_problem(problem, evaluation_function, minimize),
            favor_less_complex_trees,
        )

        self.grammar = grammar
        self.representation = representation
        self.evaluation_function = evaluation_function
        self.random = randomSource(seed)
        self.seed = seed
        self.population_size = population_size
        self.max_depth = max_depth
        self.favor_less_complex_trees = favor_less_complex_trees
        self.minimize = minimize
        self.number_of_generations = number_of_generations
        self.force_individual = force_individual
        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []
        if save_to_csv:
            c = CSVCallback(
                save_to_csv,
                save_genotype_as_string=save_genotype_as_string,
            )
            self.callbacks.append(c)

    def evolve(self, verbose=1):
        self.callbacks = [
            cb
            for cb in self.callbacks
            if type(cb)
            not in [
                type(DebugCallback()),
                type(PrintBestCallback()),
                type(ProgressCallback()),
            ]
        ]
        if verbose > 2:
            self.callbacks.append(DebugCallback())
        if verbose > 1:
            self.callbacks.append(PrintBestCallback())
        if verbose > 0:
            self.callbacks.append(ProgressCallback())

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
            start = time.time()
            population = [
                self.create_individual(self.max_depth)
                for _ in range(self.population_size)
            ]
            for ind in population:
                f = self.keyfitness()(ind)
                if (best == None) or (f < best):
                    best = f
                    best_ind = ind

            time_gen = time.time() - start
            for cb in self.callbacks:
                cb.process_iteration(gen + 1, population, time=time_gen, gp=self)

        for cb in self.callbacks:
            cb.end_evolution()
        return (
            best_ind,
            self.evaluate(best_ind),
            self.representation.genotype_to_phenotype(
                self.grammar,
                best_ind.genotype,
            ),
        )
