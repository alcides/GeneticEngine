from __future__ import annotations

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
from geneticengine.algorithms.gp.gp import GP
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


class GPFriendly(GP):
    """This is the user-friendly version of GP."""

    def __init__(
        self,
        grammar: Grammar,
        problem: Problem = None,
        evaluation_function: Callable[
            [Any],
            float,
        ] = None,  # DEPRECATE in the next version
        minimize: bool = False,  # DEPRECATE in the next version
        target_fitness: float | None = None,  # DEPRECATE in the next version
        favor_less_deep_trees: bool = True,  # DEPRECATE in the next version
        randomSource: Callable[[int], RandomSource] = RandomSource,
        population_size: int = 200,
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
        seed: int = 123,
        # -----
        timer_stop_criteria: bool = False,  # TODO: This should later be generic
        timer_limit: int = 60,
        # -----
        save_to_csv: str = None,
        save_genotype_as_string: bool = True,
        test_data: Callable[
            [Any],
            float,
        ] = None,  # TODO: Should be part of Problem Class  [LEON]
        only_record_best_inds: bool = True,
        # -----
        callbacks: list[Callback] = None,
        **kwargs,
    ):

        representation = kwargs.get("representation", treebased_representation)
        self.problem: Problem = wrap_depth(
            process_problem(problem, evaluation_function, minimize, target_fitness),
            favor_less_deep_trees,
        )

        GP.__init__(self, grammar, representation, problem, randomSource)
