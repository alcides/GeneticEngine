from __future__ import annotations

from typing import Any
from typing import Callable
from typing import TypeVar

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.callbacks.callback import DebugCallback
from geneticengine.algorithms.callbacks.callback import PrintBestCallback
from geneticengine.algorithms.callbacks.callback import ProgressCallback
from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.operators.combinators import ParallelStep
from geneticengine.algorithms.gp.operators.combinators import SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.initializers import FullInitializer
from geneticengine.algorithms.gp.operators.initializers import GrowInitializer
from geneticengine.algorithms.gp.operators.initializers import (
    InjectInitialPopulationWrapper,
)
from geneticengine.algorithms.gp.operators.initializers import (
    RampedHalfAndHalfInitializer,
)
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.pge import PGECallback
from geneticengine.algorithms.gp.operators.selection import LexicaseSelection
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.algorithms.gp.operators.stop import TimeStoppingCriterium
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.algorithms.gp.structure import StoppingCriterium
from geneticengine.algorithms.heuristics import Heuristics
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import FitnessType
from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.problems import wrap_depth_minimization
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import relabel_nodes_of_trees
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation

P = TypeVar("P")


class GPFriendly(GP):
    """A simpler API to create GP instances.

    Defaults as given in A Field Guide to GP, p.17, by Poli and Mcphee:
    Args:
        grammar (Grammar): The grammar used to guide the search.
        representation (Representation): The individual representation used by the GP program. The default is treebased_representation.
        problem (Problem): The problem we are solving. Either a SingleObjectiveProblem or a MultiObjectiveProblem.
        evaluation_function (Callable[[Any], float]): The fitness function. Should take in any valid individual and return a float. The default is that the higher the fitness, the more applicable is the solution to the problem. Turn on the parameter minimize to switch it around.
        minimize (bool): When switch on, the fitness function is reversed, so that a higher result from the fitness function corresponds to a less fit solution (default = False).
        target_fitness (Optional[float]): Sets a target fitness. When this fitness is reached, the algorithm stops running (default = None).
        favor_less_deep_trees (bool): If set to True, this gives a tiny penalty to deeper trees to favor simpler trees (default = False).
        randomSource (Callable[[int], RandomSource]): The random source function used by the program. Should take in an integer, representing the seed, and return a RandomSource.
        population_size (int): The population size (default = 200). Apart from the first generation, each generation the population is made up of the elites, novelties, and transformed individuals from the previous generation. Note that population_size > (n_elites + n_novelties + 1) must hold.
        n_elites (int): Number of elites, i.e. the number of best individuals that are preserved every generation (default = 5).
        n_novelties (int): Number of novelties, i.e. the number of newly generated individuals added to the population each generation. (default = 10).
        number_of_generations (int): Number of generations (default = 100).
        max_depth (int): The maximum depth a tree can have (default = 15).
        max_init_depth (int): The maximum depth a tree can have in the initialisation population (default = 15).
        selection_method (Tuple[str, int]): Allows the user to define the method to choose individuals for the next population (default = ("tournament", 5)).
        probability_mutation (float): probability that an individual is mutated (default = 0.01).
        probability_crossover (float): probability that an individual is chosen for cross-over (default = 0.9).
        either_mut_or_cro (float | None): Switch evolution style to do either a mutation or a crossover. The given float defines the chance of a mutation. Otherwise a crossover is performed. (default = None),
        hill_climbing (bool): Allows the user to change the standard mutation operations to the hill-climbing mutation operation, in which an individual is mutated to 5 different new individuals, after which the best is chosen to survive (default = False).
        specific_type_mutation (type): Specify a type that is given preference when mutation occurs (default = None),
        specific_type_crossover (type): Specify a type that is given preference when crossover occurs (default = None),
        depth_aware_mut (bool): If chosen, mutations are depth-aware, giving preference to operate on nodes closer to the root. (default = True).
        depth_aware_co (bool): If chosen, crossovers are depth-aware, giving preference to operate on nodes closer to the root. (default = True).
        force_individual (Any): Allows the incorporation of an individual in the first population (default = None).
        seed (int): The seed of the RandomSource (default = 123).
        timer_stop_criteria (bool): If set to True, the algorithm is stopped after the time limit (default = 60 seconds). Then the fittest individual is returned (default = False).
        timer_limit (int): The time limit of the timer.
        save_to_csv (str): Saves a CSV file with the details of all the individuals of all generations.
        save_genotype_as_string (bool): Turn this off if you don't want to safe all the genotypes as strings. This saves memory and a bit of time.
        test_data (Any): Give test data (format: (X_test, y_test)) to test the individuals on test data during training and save that to the csv (default = None).
        only_record_best_inds (bool): Specify whether one or all individuals are saved to the csv files (default = True).
        callbacks (List[Callback]): The callbacks to define what is done with the returned prints from the algorithm (default = []).
    """

    def __init__(
        self,
        grammar: Grammar,
        representation: type = None,
        problem: Problem = None,
        evaluation_function: Callable[
            [Any],
            float,
        ] = None,  # DEPRECATE in the next version
        minimize: bool = False,  # DEPRECATE in the next version
        target_fitness: float | None = None,  # DEPRECATE in the next version
        favor_less_deep_trees: bool = True,  # DEPRECATE in the next version
        source_generator: Callable[[int], RandomSource] = RandomSource,
        seed: int = 123,
        population_size: int = 200,
        initialization_method: str = "ramped",
        n_elites: int = 5,  # Shouldn't this be a percentage of population size?
        n_novelties: int = 10,
        number_of_generations: int = 100,
        max_depth: int = 15,
        # now based on depth, maybe on number of nodes?
        # selection-method is a tuple because tournament selection needs to receive the tournament size
        # but lexicase does not need a tuple
        selection_method: tuple[str, int] = ("tournament", 5),
        # -----
        # As given in A Field Guide to GP, p.17, by Poli and Mcphee
        probability_mutation: float = 0.01,
        probability_crossover: float = 0.9,
        either_mut_or_cro: float | None = None,
        hill_climbing: bool = False,
        specific_type_mutation: type = None,
        specific_type_crossover: type = None,
        depth_aware_mut: bool = False,
        depth_aware_co: bool = False,
        # -----
        force_individual: Any = None,
        # -----
        timer_stop_criteria: bool = False,  # TODO: This should later be generic
        timer_limit: int = 60,
        # ---
        evolve_grammar: bool = False,
        evolve_learning_rate: float = 0.01,
        # -----
        save_to_csv: str = None,
        save_genotype_as_string: bool = True,
        test_data: Callable[
            [Any],
            float,
        ] = None,  # TODO: Should be part of Problem Class  [LEON]
        only_record_best_inds: bool = True,
        # -----
        verbose=1,
        base_callbacks: list[Callback] = None,
        **kwargs,
    ):

        representation_class = representation or TreeBasedRepresentation
        representation_instance: Representation = representation_class(
            grammar,
            max_depth,
        )

        processed_problem: Problem = self.wrap_depth(
            self.process_problem(
                problem,
                evaluation_function,
                minimize,
                target_fitness,
            ),
            favor_less_deep_trees,
        )
        random_source = source_generator(seed)

        population_initializer: PopulationInitializer = {
            "grow": GrowInitializer,
            "full": FullInitializer,
            "ramped": RampedHalfAndHalfInitializer,
        }[initialization_method]()
        if force_individual:
            population_initializer = InjectInitialPopulationWrapper(
                [representation_instance.phenotype_to_genotype(force_individual)],
                population_initializer,
            )

        step: GeneticStep
        if either_mut_or_cro is not None:
            mutation_step = GenericMutationStep(
                1,
                specific_type=specific_type_mutation,
                depth_aware_mut=depth_aware_mut,
            )
            crossover_step = GenericCrossoverStep(
                1,
                specific_type=specific_type_crossover,
                depth_aware_co=depth_aware_co,
            )
            step = ParallelStep(
                [mutation_step, crossover_step],
                [either_mut_or_cro, 1 - either_mut_or_cro],
            )
        else:
            mutation_step = GenericMutationStep(
                probability_mutation,
                specific_type=specific_type_mutation,
                depth_aware_mut=depth_aware_mut,
            )
            crossover_step = GenericCrossoverStep(
                probability_crossover,
                specific_type=specific_type_crossover,
                depth_aware_co=depth_aware_co,
            )
            step = SequenceStep(mutation_step, crossover_step)

        current_population_size = population_size
        if n_elites > 0:
            current_population_size -= n_elites
            step = ParallelStep([ElitismStep()], [n_elites, current_population_size])

        if n_novelties > 0:
            current_population_size -= n_novelties
            step = ParallelStep([NoveltyStep()], [n_novelties, current_population_size])

        selection_step: GeneticStep
        if selection_method[0] == "tournament":
            selection_step = TournamentSelection(selection_method[1])
        elif selection_method[0] == "lexicase":
            selection_step = LexicaseSelection()
        else:
            raise ValueError(
                f"selection_method ({selection_method}) requires either tournament or lexicase",
            )
        step = SequenceStep(selection_step, step)

        stopping_criterium: StoppingCriterium
        if timer_stop_criteria:
            stopping_criterium = TimeStoppingCriterium(timer_limit)
        else:
            stopping_criterium = GenerationStoppingCriterium(number_of_generations)

        callbacks: list[Callback] = []
        callbacks.extend(base_callbacks or [])

        if evolve_grammar:
            callbacks.append(PGECallback(evolve_learning_rate))

        if verbose > 2:
            self.callbacks.append(DebugCallback())
        if verbose > 1:
            self.callbacks.append(PrintBestCallback())
        if verbose > 0:
            self.callbacks.append(ProgressCallback())

        if save_to_csv:
            self.test_data = test_data
            if self.test_data:

                def test_evaluate(individual: Individual) -> float:
                    phenotype = representation_instance.genotype_to_phenotype(
                        individual.genotype,
                    )
                    test_fitness = test_data(phenotype)  # type: ignore
                    return test_fitness

                self.test_data = test_evaluate

            c = CSVCallback(
                save_to_csv,
                test_data=self.test_data,
                only_record_best_ind=only_record_best_inds,
                save_genotype_as_string=save_genotype_as_string,
            )
            self.callbacks.append(c)

        GP.__init__(
            self,
            representation_instance,
            processed_problem,
            random_source,
            population_size,
            population_initializer,
            step,
            stopping_criterium,
            callbacks,
        )

    def process_problem(
        self,
        problem: Problem | None,
        evaluation_function: Callable[[P], float] = None,
        minimize: bool = False,
        target_fitness: float | None = None,
    ) -> Problem:
        """This function is a placeholder until we deprecate all the old usage
        of GP class."""
        if problem:
            return problem
        elif isinstance(minimize, list) and evaluation_function:
            return MultiObjectiveProblem(minimize, evaluation_function)
        elif isinstance(minimize, bool) and evaluation_function:
            return SingleObjectiveProblem(minimize, evaluation_function, target_fitness)
        else:
            raise NotImplementedError(
                "This combination of parameters to define the problem is not valid",
            )

    def wrap_depth(self, p: Problem, favor_less_deep_trees: bool = False):
        if isinstance(p, SingleObjectiveProblem):
            if favor_less_deep_trees:
                return wrap_depth_minimization(p)
            else:
                return p
        else:
            assert isinstance(p, MultiObjectiveProblem)
            return p