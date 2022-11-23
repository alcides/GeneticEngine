from __future__ import annotations

import time
from typing import Any
from typing import Callable
from typing import NoReturn

import geneticengine.algorithms.gp.generation_steps.cross_over as cross_over
import geneticengine.algorithms.gp.generation_steps.mutation as mutation
import geneticengine.algorithms.gp.generation_steps.selection as selection
from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.callbacks.callback import DebugCallback
from geneticengine.algorithms.callbacks.callback import PrintBestCallback
from geneticengine.algorithms.callbacks.callback import ProgressCallback
from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.heuristics import Heuristics
from geneticengine.core.grammar import EvolveGrammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import FitnessType
from geneticengine.core.problems import Problem
from geneticengine.core.problems import process_problem
from geneticengine.core.problems import wrap_depth
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import relabel_nodes_of_trees
from geneticengine.core.representations.tree.treebased import treebased_representation


class GP(Heuristics):
    """Represents the Genetic Programming algorithm. Defaults as given in A
    Field Guide to GP, p.17, by Poli and Mcphee:

    Args:
        grammar (Grammar): The grammar used to guide the search.
        representation (Representation): The individual representation used by the GP program. The default is treebased_representation.
        problem (Problem): The problem we are solving. Either a SingleObjectiveProblem or a MultiObjectiveProblem.
        evaluation_function (Callable[[Any], float]): The fitness function. Should take in any valid individual and return a float. The default is that the higher the fitness, the more applicable is the solution to the problem. Turn on the parameter minimize to switch it around.
        minimize (bool): When switch on, the fitness function is reversed, so that a higher result from the fitness function corresponds to a less fit solution (default = False).
        target_fitness (Optional[float]): Sets a target fitness. When this fitness is reached, the algorithm stops running (default = None).
        favor_less_complex_trees (bool): If set to True, this gives a tiny penalty to more complex (with more nodes) trees to favor simpler trees (default = False).
        randomSource (Callable[[int], RandomSource]): The random source function used by the program. Should take in an integer, representing the seed, and return a RandomSource.
        population_size (int): The population size (default = 200). Apart from the first generation, each generation the population is made up of the elites, novelties, and transformed individuals from the previous generation. Note that population_size > (n_elites + n_novelties + 1) must hold.
        n_elites (int): Number of elites, i.e. the number of best individuals that are preserved every generation (default = 5).
        n_novelties (int): Number of novelties, i.e. the number of newly generated individuals added to the population each generation. (default = 10).
        number_of_generations (int): Number of generations (default = 100).
        max_depth (int): The maximum depth a tree can have (default = 15).
        max_init_depth (int): The maximum depth a tree can have in the initialisation population. Currently only working for tree-based representation and dynamic SGE (default = max_depth).
        min_depth (int): The minimum depth a tree can have (default = None).
        min_init_depth (int): The minimum depth a tree can have in the initialisation population. Only relevant when using the Random_Production initialization method (default = None).
        selection_method (Tuple[str, int]): Allows the user to define the method to choose individuals for the next population (default = ("tournament", 5)).
        ramped_half_and_half (bool): Specify whether you want the population to be initialized ramped half and half. Currently only working for treebased representation (default = True).
        evolve_grammar (EvolveGrammar): You can evolve the grammar throughout the generations by assigning a EvolveGrammar to this parameter (default: None).

        probability_mutation (float): probability that an individual is mutated (default = 0.01).
        probability_crossover (float): probability that an individual is chosen for cross-over (default = 0.9).
        either_mut_or_cro (float | None): Switch evolution style to do either a mutation or a crossover. The given float defines the chance of a mutation. Otherwise a crossover is performed. (default = None),
        hill_climbing (bool): Allows the user to change the standard mutation operations to the hill-climbing mutation operation, in which an individual is mutated to 5 different new individuals, after which the best is chosen to survive (default = False).
        specific_type_mutation (type): Specify a type that is given preference when mutation occurs (default = None),
        specific_type_crossover (type): Specify a type that is given preference when crossover occurs (default = None),
        depth_aware_mut (bool): If chosen, mutations are depth-aware, giving preference to operate on nodes closer to the root. (default = True).
        depth_aware_co (bool): If chosen, crossovers are depth-aware, giving preference to operate on nodes closer to the root. (default = True).
        cross_over_return_one_individual (bool): If chosen, crossovers return only one individual, and the other is discarded. Notice that Genetic Engine slows down through this. (default = False)

        force_individual (Any): Allows the incorporation of an individual in the first population (default = None).
        seed (int): The seed of the RandomSource (default = 123).

        timer_stop_criteria (bool): If set to True, the algorithm is stopped after the time limit (default = 60 seconds). Then the fittest individual is returned (default = False).
        timer_limit (int): The time limit of the timer.

        save_to_csv (CSVCallback): Saves a CSV file that can be personalized in the CSVCallback class specified (default = None).

        callbacks (List[Callback]): The callbacks to define what is done with the returned prints from the algorithm (default = []).
    """

    # reason for union with noreturn in evaluation function, elitism and elitism: https://stackoverflow.com/questions/51811024/mypy-type-checking-on-callable-thinks-that-member-variable-is-a-method
    grammar: Grammar
    representation: Representation[Any]
    problem: Problem
    evaluation_function: NoReturn | Callable[[Any], float]
    random: RandomSource
    population_size: int
    elitism: (
        NoReturn
        | Callable[
            [
                list[Individual],
                Problem,
                Callable[[Problem, list[Individual]], Individual],
                Callable[[Individual], float | list[float]],
            ],
            list[Individual],
        ]
    )
    mutation: (NoReturn | Callable[[Individual], Individual])
    max_depth: int
    novelty: NoReturn | Callable[[int], list[Individual]]
    final_population: list[Individual]
    callbacks: list[Callback]

    def __init__(
        self,
        grammar: Grammar,
        representation: Representation = treebased_representation,
        # These classes should be replaced by the problem alone.
        problem: Problem | None = None,
        evaluation_function: Callable[
            [Any],
            float,
        ] | None = None,  # DEPRECATE in the next version
        minimize: bool = False,  # DEPRECATE in the next version
        target_fitness: float | None = None,  # DEPRECATE in the next version
        evolve_grammar: EvolveGrammar | None = None,
        favor_less_complex_trees: bool = True,  # DEPRECATE in the next version
        randomSource: Callable[[int], RandomSource] = RandomSource,
        population_size: int = 200,
        n_elites: int = 5,
        n_novelties: int = 10,
        number_of_generations: int = 100,
        max_depth: int = 15,
        max_init_depth: int | None = None,
        min_depth: int | None = None,
        min_init_depth: int | None = None,
        selection_method: tuple[str, int] = ("tournament", 5),
        ramped_half_and_half: bool = False,
        # -----
        # As given in A Field Guide to GP, p.17, by Poli and Mcphee
        probability_mutation: float = 0.01,
        probability_crossover: float = 0.9,
        either_mut_or_cro: float | None = None,
        hill_climbing: bool = False,
        specific_type_mutation: type | None = None,
        specific_type_crossover: type | None = None,
        depth_aware_mut: bool = False,
        depth_aware_co: bool = False,
        cross_over_return_one_individual: bool = False,
        # -----
        force_individual: Any = None,
        seed: int = 123,
        # -----
        timer_stop_criteria: bool = False,
        timer_limit: int = 60,
        # -----
        save_to_csv: CSVCallback | None = None,
        # -----
        callbacks: list[Callback] | None = None,
    ):
        assert population_size > (n_elites + n_novelties + 1)

        self.problem: Problem = wrap_depth(
            process_problem(problem, evaluation_function, minimize, target_fitness),
            favor_less_complex_trees,
        )

        self.grammar = grammar
        self.representation = representation
        self.random = randomSource(seed)
        self.seed = seed
        self.population_size = population_size
        self.elitism = selection.create_elitism(n_elites)
        self.max_depth = max_depth
        if max_init_depth:
            assert max_init_depth <= self.max_depth
            self.max_init_depth = max_init_depth
        else:
            self.max_init_depth = self.max_depth
        if min_depth:
            assert min_depth <= self.max_depth
        self.min_depth = min_depth
        if min_init_depth:
            assert min_init_depth <= self.max_init_depth
            self.min_init_depth = min_init_depth
        else:
            self.min_init_depth = self.min_depth # type: ignore
        self.evolve_grammar = evolve_grammar
        self.favor_less_complex_trees = favor_less_complex_trees
        self.novelty = selection.create_novelties(
            self.create_individual,
            max_depth=max_depth,
        )
        self.timer_stop_criteria = timer_stop_criteria
        self.timer_limit = timer_limit
        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []
        if hill_climbing:
            self.mutation = mutation.create_hill_climbing_mutation(
                self.random,
                self.representation,
                self.grammar,
                max_depth,
                self.keyfitness(),
                5,
                specific_type=specific_type_mutation,
                depth_aware_mut=depth_aware_mut,
            )
        else:
            self.mutation = mutation.create_mutation(
                self.random,
                self.representation,
                self.grammar,
                max_depth,
                specific_type=specific_type_mutation,
                depth_aware_mut=depth_aware_mut,
            )
        self.cross_over = cross_over.create_cross_over(
            self.random,
            self.representation,
            self.grammar,
            max_depth,
            specific_type=specific_type_crossover,
            depth_aware_co=depth_aware_co,
        )
        self.n_novelties = n_novelties
        self.number_of_generations = number_of_generations
        self.probability_mutation = probability_mutation
        self.probability_crossover = probability_crossover
        self.either_mut_or_cro = either_mut_or_cro
        self.cross_over_return_one_individual = cross_over_return_one_individual
        if selection_method[0] == "tournament":
            self.selection = selection.create_tournament(
                selection_method[1],
                self.problem,
            )
        elif selection_method[0] == "lexicase":
            self.selection = selection.create_lexicase(
                self.problem,
            )
        else:
            self.selection = lambda r, ls, n: [x for x in ls[:n]]
        self.force_individual = force_individual
        self.ramped_half_and_half = ramped_half_and_half

        if save_to_csv:
            self.callbacks.append(save_to_csv)

    def evolve(self, verbose=1) -> tuple[Individual, FitnessType, Any]:
        """The main function of the GP object. This function runs the GP
        algorithm over the set number of generations, evolving better
        solutions.

        Args:
            verbose (int): Sets the verbose level of the function (0: no prints, 1: print progress, or 2: print the best individual in each generation).

        Returns a tuple with the following arguments:
            individual (Individual): The fittest individual after the algorithm has finished.
            fitness (float): The fitness of above individual.
            phenotype (Any): The phenotype of the best individual.
        """
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

        # TODO: This is not ramped half and half
        population = self.init_population(self.ramped_half_and_half)
        if self.force_individual is not None:
            population[0] = Individual(
                genotype=relabel_nodes_of_trees(
                    self.force_individual,
                    self.grammar,
                ),
                fitness=None,
            )
        gen = 0
        start = time.time()

        if self.evolve_grammar:
            best_overall = True

        while (not self.timer_stop_criteria and gen < self.number_of_generations) or (
            self.timer_stop_criteria and (time.time() - start) < self.timer_limit
        ):

            npop = self.novelty(self.n_novelties)
            elites = self.elitism(
                population,
                self.problem,
                self.get_best_individual,
                self.evaluate,
            )
            npop.extend(elites)
            spotsLeft = self.population_size - len(npop)
            while spotsLeft > 0:
                if self.either_mut_or_cro:
                    if self.random.random_float(0, 1) < self.either_mut_or_cro:
                        candidate = self.selection(self.random, population, 1)
                        p1 = self.mutation(candidate[0])
                        npop.append(p1)
                        spotsLeft -= 1
                    else:
                        candidates = self.selection(self.random, population, 2)
                        (p1, p2) = self.cross_over(candidates[0], candidates[1])
                        npop.append(p1)
                        if self.cross_over_return_one_individual:
                            spotsLeft -= 1
                        else:
                            npop.append(p2)
                            spotsLeft -= 2
                else:
                    candidates = self.selection(self.random, population, 2)
                    (p1, p2) = candidates[0], candidates[1]
                    if self.random.randint(0, 100) < self.probability_crossover * 100:
                        (p1, p2) = self.cross_over(p1, p2)
                    if self.random.randint(0, 100) < self.probability_mutation * 100:
                        p1 = self.mutation(p1)
                    npop.append(p1)
                    if self.cross_over_return_one_individual:
                        spotsLeft -= 1
                    else:
                        if self.random.randint(0, 100) < self.probability_mutation * 100:
                            p2 = self.mutation(p2)
                        npop.append(p2)
                        spotsLeft -= 2

            population = npop

            best_individual = self.get_best_individual(self.problem, population)

            time_gen = time.time() - start
            for cb in self.callbacks:
                cb.process_iteration(gen + 1, population, time=time_gen, gp=self)

            if self.evolve_grammar:
                if best_overall:
                    inds = elites
                else:
                    for e in elites:
                        npop.remove(e)
                    inds = npop
                best = self.get_best_individual(self.problem, inds)
                prob = best.production_probabilities(
                    self.representation.genotype_to_phenotype,
                    self.grammar,
                )
                self.grammar = self.grammar.update_weights(
                    self.evolve_grammar.learning_rate,
                    prob,
                )
                best_overall = not best_overall
            gen += 1
        self.final_population = population

        for cb in self.callbacks:
            cb.end_evolution()
        return (
            best_individual,
            self.evaluate(best_individual),
            self.representation.genotype_to_phenotype(
                self.grammar,
                best_individual.genotype,
            ),
        )

    def init_population(self, ramped_half_and_half):
        
        def create_ind_w_restrictions(max_depth, min_init, max_init):
            def check_depths(ind: Individual):
                phenotype = self.representation.genotype_to_phenotype(self.grammar, ind.genotype)
                conforms_min_depth = True if not min_init else (phenotype.gengy_distance_to_term >= min_init)
                conforms_max_depth = True if not max_init else (phenotype.gengy_distance_to_term <= max_init)
                if not (conforms_min_depth and conforms_max_depth):
                    print(phenotype.gengy_distance_to_term)
                return conforms_min_depth and conforms_max_depth
            
            conforms_depth = False
            while not conforms_depth:
                self.representation.method.min_depth = self.min_init_depth
                ind = self.create_individual(max_init)
                self.representation.method.min_depth = self.min_depth # Used for genotype-to-phenotype mapping
                self.representation.depth = self.max_depth # Used for genotype-to-phenotype mapping
                conforms_depth = check_depths(ind)
            return ind
        
        if ramped_half_and_half:
            n_not_ramped = int(self.population_size / 2)
            n_ramped = self.population_size - n_not_ramped 
            pop_ramped = [
                create_ind_w_restrictions(self.max_depth, min_init=self.min_init_depth, max_init=max((i % self.max_init_depth) + 1, self.grammar.get_min_tree_depth()))
                for i in range(n_ramped)
            ]
            pop_not_ramped = [
                create_ind_w_restrictions(self.max_depth, min_init=self.min_init_depth, max_init=self.max_init_depth)
                for _ in range(n_not_ramped)
            ]
                
            pop = pop_ramped + pop_not_ramped
        else:
            pop = [
                create_ind_w_restrictions(self.max_depth, min_init=self.min_init_depth, max_init=self.max_init_depth)
                for _ in range(self.population_size)
            ]
        
        self.representation.depth = self.max_depth
        self.representation.method.min_depth = self.min_depth
        return pop
        
