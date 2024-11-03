from __future__ import annotations

from typing import Any, Optional, overload
from typing import Callable
from typing import TypeVar

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.combinators import ExclusiveParallelStep
from geneticengine.algorithms.gp.operators.combinators import ParallelStep
from geneticengine.algorithms.gp.operators.combinators import SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import LexicaseSelection
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.evaluation.budget import EvaluationBudget, TimeBudget
from geneticengine.evaluation.parallel import ParallelEvaluator
from geneticengine.evaluation.recorder import CSVSearchRecorder, SearchRecorder
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.grammar.grammar import Grammar
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import MultiObjectiveProblem
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.grammatical_evolution.dynamic_structured_ge import (
    DynamicStructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.representations.grammatical_evolution.ge import GrammaticalEvolutionRepresentation
from geneticengine.representations.grammatical_evolution.structured_ge import (
    StructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.representations.stackgggp import StackBasedGGGPRepresentation
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.operators import (
    GrowInitializer,
    InjectInitialPopulationWrapper,
)
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import Individual

P = TypeVar("P")


class SimpleGP:
    """A simple API to create a new Genetic Programming instance.

    Args:
        fitness_function (Callable[[Any]->float|list[float]]): the fitness function that assesses the quality of individuals
        minimize (bool): whether the fitness function should be minimized (default=False)

        grammar (Grammar): The grammar used to guide the search.
        representation (Representation): The individual representation used by the GP program. The default is
            TreeBasedRepresentation.
        problem (Problem): The problem we are solving. Either a SingleObjectiveProblem or a MultiObjectiveProblem.

        TODO: The rest of parameters
    """

    def __init__(
        self,
        # Problem:
        fitness_function: Callable[[Any], Any],
        grammar: Grammar,
        minimize: bool | list[bool] = False,
        # Representation + Grammar
        representation: str = "treebased",
        max_depth: int = 15,
        # Budget
        target_fitness: list[float] | float | None = None,
        max_time: Optional[float] = None,
        max_evaluations: Optional[int] = None,
        # Tracker
        csv_output: str | None = None,
        csv_extra_fields: dict[str, Callable[[Any], str]] | None = None,
        only_record_best_individuals: bool = True,
        parallel_evaluation=False,
        # RNG
        seed: int = 0,
        # GP Specific:
        population_size: int = 200,
        elitism: int = 10,
        novelty: int = 10,
        initial_population: list[Any] | None = None,
        mutation_probability: float = 0.01,
        crossover_probability: float = 0.9,
        selection_method: tuple[str, int | bool] = ("tournament", 5),
    ):
        self.random = NativeRandomSource(seed)
        self.problem = self.process_problem(fitness_function, minimize, target_fitness)
        budget = self.build_budget(max_time, max_evaluations)
        representation_internal = self.process_representation(representation, grammar, max_depth)
        population_initializer = self.process_population_initializer(initial_population)

        step = self.build_step(
            population_size,
            elitism,
            novelty,
            mutation_probability,
            crossover_probability,
            selection_method,
        )
        recorder = self.build_recorder(
            self.problem,
            csv_output,
            only_record_best_individuals,
            parallel_evaluation,
            csv_extra_fields,
        )
        self.gp = GeneticProgramming(
            self.problem,
            budget=budget,
            random=self.random,
            representation=representation_internal,
            population_size=population_size,
            population_initializer=population_initializer,
            step=step,
            tracker=recorder,
        )

    def search(self) -> list[Individual] | None:
        return self.gp.search()

    @overload
    def process_problem(
        self,
        fitness_function: Callable[[P], float],
        minimize: bool | list[bool],
        target_fitness: list[float] | float | None,
    ) -> Problem: ...

    @overload
    def process_problem(
        self,
        fitness_function: Callable[[P], list[float]],
        minimize: list[bool],
        target_fitness: Optional[list[float]],
    ) -> Problem: ...

    def process_problem(self, fitness_function, minimize=False, target_fitness=None) -> Problem:
        if isinstance(minimize, list):
            return MultiObjectiveProblem(fitness_function, minimize, target_fitness)
        else:
            return SingleObjectiveProblem(fitness_function, minimize, target_fitness)

    def process_representation(self, representation: str, grammar: Grammar, max_depth: int):
        representation_class = {
            "treebased": TreeBasedRepresentation,
            "ge": GrammaticalEvolutionRepresentation,
            "sge": StructuredGrammaticalEvolutionRepresentation,
            "dsge": DynamicStructuredGrammaticalEvolutionRepresentation,
            "stack": StackBasedGGGPRepresentation,
        }[representation]

        decider = MaxDepthDecider(self.random, grammar, max_depth)

        return representation_class(grammar=grammar, decider=decider)

    def build_budget(self, max_time, max_evaluations):
        if max_time is None and max_evaluations is None:
            assert False, "You have to define wither the max_time or max_evaluations"
        elif max_time is not None:
            return TimeBudget(max_time)
        else:
            return EvaluationBudget(max_evaluations)

    def process_population_initializer(self, initial_population: list[Any] | None = None):
        if initial_population:
            return InjectInitialPopulationWrapper(
                [genotype for genotype in initial_population],
                GrowInitializer(),
            )
        else:
            return GrowInitializer()

    def build_step(
        self,
        population_size: int,
        elitism: int,
        novelty: int,
        mutation_probability: float,
        crossover_probability: float,
        selection_method: tuple[str, int | bool],
    ):
        step: GeneticStep

        step = ExclusiveParallelStep(
            [GenericMutationStep(mutation_probability), GenericCrossoverStep(crossover_probability)],
        )

        selection_step: GeneticStep
        if selection_method[0] == "tournament":
            selection_step = TournamentSelection(selection_method[1])
        elif selection_method[0] == "lexicase":
            ep: bool = bool(selection_method[1]) if len(selection_method) > 1 else False
            selection_step = LexicaseSelection(epsilon=ep)
        else:
            raise ValueError(
                f"selection_method ({selection_method}) requires either tournament or lexicase",
            )
        step = SequenceStep(selection_step, step)

        assert elitism + novelty <= population_size
        step = ParallelStep(
            [ElitismStep(), NoveltyStep(), step],
            [elitism, novelty, population_size - novelty - elitism],
        )
        return step

    def build_recorder(
        self,
        problem: Problem,
        csv_output: str | None,
        only_record_best_individuals: bool = True,
        parallel_evaluation: bool = False,
        csv_extra_fields: dict[str, Callable[[Any], str]] | None = None,
    ):
        recorders: list[SearchRecorder] = []
        if csv_extra_fields:
            csv_extra_fields2 = {cb: lambda t, i, p: csv_extra_fields[cb](i.get_phenotype()) for cb in csv_extra_fields}
        else:
            csv_extra_fields2 = None
        if csv_output:
            recorder = CSVSearchRecorder(
                csv_output,
                problem,
                only_record_best_individuals=only_record_best_individuals,
                extra_fields=csv_extra_fields2,
            )
            recorders.append(recorder)
        ev = SequentialEvaluator() if not parallel_evaluation else ParallelEvaluator()
        if problem.number_of_objectives() == 1:
            return ProgressTracker(problem=problem, evaluator=ev, recorders=recorders)
        else:
            return ProgressTracker(problem=problem, evaluator=ev, recorders=recorders)

    def get_problem(self):
        return self.problem
