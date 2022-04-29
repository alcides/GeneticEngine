from __future__ import annotations

import time
from typing import Any
from typing import Callable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Union

import geneticengine.algorithms.gp.generation_steps.cross_over as cross_over
import geneticengine.algorithms.gp.generation_steps.mutation as mutation
import geneticengine.algorithms.gp.generation_steps.selection as selection
from geneticengine.algorithms.gp.callback import Callback
from geneticengine.algorithms.gp.callback import DebugCallback
from geneticengine.algorithms.gp.callback import PrintBestCallback
from geneticengine.algorithms.gp.callback import ProgressCallback
from geneticengine.algorithms.gp.csv_callback import CSVCallback
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import relabel_nodes_of_trees
from geneticengine.core.representations.tree.treebased import treebased_representation


class GP:
    """
    Genetic Programming object. Main attribute: evolve

    Parameters:
        - grammar (Grammar): The grammar used to guide the search.
        - evaluation_function (Callable[[Any], float]): The fitness function. Should take in any valid individual and return a float. The default is that the higher the fitness, the more applicable is the solution to the problem. Turn on the parameter minimize to switch it around.
        - minimize (bool): When switch on, the fitness function is reversed, so that a higher result from the fitness function corresponds to a less fit solution (default = False).
        - representation (Representation): The individual representation used by the GP program. The default is treebased_representation.
        - randomSource (Callable[[int], RandomSource]): The random source function used by the program. Should take in an integer, representing the seed, and return a RandomSource.
        - seed (int): The seed of the RandomSource (default = 123).
        - population_size (int): The population size (default = 200). Apart from the first generation, each generation the population is made up of the elites, novelties, and transformed individuals from the previous generation. Note that population_size > (n_elites + n_novelties + 1) must hold.
        - n_elites (int): Number of elites, i.e. the number of best individuals that are preserved every generation (default = 5).
        - n_novelties (int): Number of novelties, i.e. the number of newly generated individuals added to the population each generation. (default = 10).
        - number_of_generations (int): Number of generations (default = 100).
        - max_depth (int): The maximum depth a tree can have (default = 15).
        - favor_less_deep_trees (bool): If set to True, this gives a tiny penalty to deeper trees to favor simpler trees (default = False).
        - selection_method (Tuple[str, int]): Allows the user to define the method to choose individuals for the next population (default = ("tournament", 5)).
        - hill_climbing (bool): Allows the user to change the standard mutation operations to the hill-climbing mutation operation, in which an individual is mutated to 5 different new individuals, after which the best is chosen to survive (default = False).
        - target_fitness (Optional[float]): Sets a target fitness. When this fitness is reached, the algorithm stops running (default = None).
        - force_individual (Any): Allows the incorporation of an individual in the first population (default = None).
        - timer_stop_criteria (bool): If set to True, the algorithm is stopped after the time limit (60 seconds). Then the fittest individual is returned (default = False).
        - save_to_csv (str): Saves a CSV file with the details of all the individuals of all generations.
        - callbacks (List[Callback]): The callbacks to define what is done with the returned prints from the algorithm (default = []).
        -----
        Default as given in A Field Guide to GP, p.17, by Poli and Mcphee:
        - probability_mutation (float): probability that an individual is mutated (default = 0.01).
        - probability_crossover (float): probability that an individual is chosen for cross-over (default = 0.9).
        -----

    """

    # reason for union with noreturn in evaluation function, elitism and elitism: https://stackoverflow.com/questions/51811024/mypy-type-checking-on-callable-thinks-that-member-variable-is-a-method
    grammar: Grammar
    representation: Representation[Any]
    evaluation_function: NoReturn | Callable[[Any], float]
    random: RandomSource
    population_size: int
    elitism: (
        NoReturn
        | Callable[
            [list[Individual], Callable[[Individual], float]],
            list[Individual],
        ]
    )
    mutation: (NoReturn | Callable[[Individual], Individual])
    max_depth: int
    novelty: NoReturn | Callable[[int], list[Individual]]
    minimize: bool
    final_population: list[Individual]
    callbacks: list[Callback]

    def __init__(
        self,
        grammar: Grammar,
        evaluation_function: Callable[[Any], float],
        representation: Representation = treebased_representation,
        randomSource: Callable[[int], RandomSource] = RandomSource,
        population_size: int = 200,
        n_elites: int = 5,  # Shouldn't this be a percentage of population size?
        n_novelties: int = 10,
        number_of_generations: int = 100,
        max_depth: int = 15,
        # now based on depth, maybe on number of nodes?
        favor_less_deep_trees: bool = False,
        selection_method: tuple[str, int] = ("tournament", 5),
        # -----
        # As given in A Field Guide to GP, p.17, by Poli and Mcphee
        probability_mutation: float = 0.01,
        probability_crossover: float = 0.9,
        # -----
        hill_climbing: bool = False,
        minimize: bool = False,
        target_fitness: float | None = None,
        force_individual: Any = None,
        seed: int = 123,
        # -----
        timer_stop_criteria: bool = False,  # TODO: This should later be generic
        timer_limit: int = 60,
        save_to_csv: str = None,
        callbacks: list[Callback] = [],
    ):
        assert population_size > (n_elites + n_novelties + 1)

        self.grammar = grammar
        self.representation = representation
        self.evaluation_function = evaluation_function
        self.random = randomSource(seed)
        self.seed = seed
        self.population_size = population_size
        self.elitism = selection.create_elitism(n_elites)
        self.max_depth = max_depth
        self.favor_less_deep_trees = favor_less_deep_trees
        self.novelty = selection.create_novelties(
            self.create_individual,
            max_depth=max_depth,
        )
        self.minimize = minimize
        self.target_fitness = target_fitness
        self.timer_stop_criteria = timer_stop_criteria
        self.timer_limit = timer_limit
        self.callbacks = callbacks
        if hill_climbing:
            self.mutation = mutation.create_hill_climbing_mutation(
                self.random,
                self.representation,
                self.grammar,
                max_depth,
                self.keyfitness(),
                5,
            )
        else:
            self.mutation = mutation.create_mutation(
                self.random,
                self.representation,
                self.grammar,
                max_depth,
            )
        self.cross_over = cross_over.create_cross_over(
            self.random,
            self.representation,
            self.grammar,
            max_depth,
        )
        self.n_novelties = n_novelties
        self.number_of_generations = number_of_generations
        self.probability_mutation = probability_mutation
        self.probability_crossover = probability_crossover
        if selection_method[0] == "tournament":
            self.selection = selection.create_tournament(
                selection_method[1],
                self.minimize,
            )
        else:
            self.selection = lambda r, ls, n: [x for x in ls[:n]]
        self.force_individual = force_individual

        if save_to_csv:
            c = CSVCallback(save_to_csv)
            self.callbacks.append(c)

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

    def fitness_correction_for_depth(self, individual: Individual) -> float:
        if (
            self.favor_less_deep_trees
        ):  # grammatical evolution does not have gengy_distance_to_term
            return individual.genotype.gengy_distance_to_term * 10**-25
        else:
            return 0

    def keyfitness(self):
        if self.minimize:
            return lambda x: self.evaluate(x) + self.fitness_correction_for_depth(x)
        else:
            return lambda x: -self.evaluate(x) - self.fitness_correction_for_depth(x)

    def evolve(self, verbose=1) -> tuple[Individual, float, Any]:
        """
        The main function of the GP object. This function runs the GP algorithm over the set number of generations, evolving better solutions

        Parameters:
            - verbose (int): Sets the verbose level of the function (0: no prints, 1: print progress, or 2: print the best individual in each generation).

        Returns a tuple with the following arguments:
            - individual (Individual): The fittest individual after the algorithm has finished.
            - fitness (float): The fitness of above individual.
            - phenotype (Any): The phenotype of the best individual.
        """
        if verbose > 2:
            self.callbacks.append(DebugCallback())
        if verbose > 1:
            self.callbacks.append(PrintBestCallback())
        if verbose > 0:
            self.callbacks.append(ProgressCallback())

        # TODO: This is not ramped half and half
        population = self.init_population()
        if self.force_individual is not None:
            population[0] = Individual(
                genotype=relabel_nodes_of_trees(
                    self.force_individual,
                    self.grammar,
                ),
                fitness=None,
            )
        population = sorted(population, key=self.keyfitness())

        gen = 0
        start = time.time()

        while (not self.timer_stop_criteria and gen < self.number_of_generations) or (
            self.timer_stop_criteria and (time.time() - start) < self.timer_limit
        ):
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

            time_gen = time.time() - start
            for cb in self.callbacks:
                cb.process_iteration(gen + 1, population, time=time_gen, gp=self)

            if (
                self.target_fitness is not None
                and population[0].fitness == self.target_fitness
            ):
                break
            gen += 1
        self.final_population = population
        for cb in self.callbacks:
            cb.end_evolution()
        return (
            population[0],
            self.evaluate(population[0]),
            self.representation.genotype_to_phenotype(
                self.grammar,
                population[0].genotype,
            ),
        )

    def init_population(self):
        return [
            self.create_individual(self.max_depth) for _ in range(self.population_size)
        ]
