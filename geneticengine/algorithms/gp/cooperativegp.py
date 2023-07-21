from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.initializers import StandardInitializer

from geneticengine.core.grammar import Grammar

from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource, Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.operators import InjectInitialPopulationWrapper
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation

a = TypeVar("a")
b = TypeVar("b")


class CooperativeGP:
    """CooperativeGP takes two representations and a function that pits two
    individuals against each other.

    Given grammar1 and grammar 2, a population of each is generated and
    evolved one after the other. The fitness function is the same for
    both evolutions, but first minimizes the function, and the second
    maximizes the fitness. It runs for a given number of iterations
    defined by the coevolutions parameter.
    """

    def __init__(
        self,
        grammar1: Grammar,
        grammar2: Grammar,
        function: Callable[[a, b], float],
        representation1: Optional[Representation] = None,
        representation2: Optional[Representation] = None,
        population1_size: int = 100,
        population2_size: int = 200,
        coevolutions: int = 1000,
        random_source: Optional[Source] = None,
        kwargs1: Optional[dict] = None,
        kwargs2: Optional[dict] = None,
    ):
        """Creates a new object to co-evolve species1 and species2.

        Arguments:
            grammar1 (Grammar): Grammar of species1
            grammar2 (Grammar): Grammar of species2
            function (Callable[[a,b], float]): Function that returns a score for a battle between one individual of species1 and another of species2
            representation1 (Representation): Representation of species1
            representation2 (Representation): Representation of species2
            population1_size (int): Population size of species1
            population2_size (int): Population size of species2
            coevolutions (int): How many iterations of a pair of evolutions
            random_source (Source): The random number generator
            kwargs1 (dict): The extra arguments for the GP object of species1
            kwargs2 (dict): The extra arguments for the GP object of species2
        """
        self.g1 = grammar1
        self.g2 = grammar2
        self.ff = {"ff": function}

        self.population1_size = population1_size
        self.population2_size = population2_size
        self.coevolutions = coevolutions
        self.representation1 = representation1 or TreeBasedRepresentation(grammar=self.g1, max_depth=10)
        self.representation2 = representation2 or TreeBasedRepresentation(grammar=self.g2, max_depth=10)
        self.kwargs1 = kwargs1 or {}
        self.kwargs2 = kwargs2 or {}
        self.random_source = random_source or RandomSource()

    def evolve(self) -> tuple[a, b]:
        @dataclass
        class Bests:
            b1: a
            b2: b

        b1: a = self.representation1.create_individual(self.random_source)  # type: ignore
        b2: b = self.representation2.create_individual(self.random_source)  # type: ignore
        self.bests = Bests(b1, b2)

        f = self.ff["ff"]
        init = StandardInitializer()

        def f1(x: a) -> float:
            return f(x, self.bests.b2)

        def f2(x: b) -> float:
            return f(self.bests.b1, x)

        p1 = SingleObjectiveProblem(fitness_function=f1, minimize=True)
        p2 = SingleObjectiveProblem(fitness_function=f2, minimize=False)

        pop1 = init.initialize(p1, self.representation1, self.random_source, self.population1_size)
        pop2 = init.initialize(p2, self.representation2, self.random_source, self.population1_size)

        for _ in range(self.coevolutions):
            # We create new problems to avoid results from previous iterations being cached.
            p1 = SingleObjectiveProblem(fitness_function=f1, minimize=True)
            p2 = SingleObjectiveProblem(fitness_function=f2, minimize=False)

            gp1 = GP(
                problem=p1,
                representation=self.representation1,
                random_source=self.random_source,
                population_size=self.population1_size,
                initializer=InjectInitialPopulationWrapper(
                    [e.get_phenotype() for e in pop1],
                    init,
                ),  # TODO: we might want to keep individuals, and not only the phenotypes.
                **self.kwargs1,
            )
            ind = gp1.evolve()
            self.bests.b1 = ind.get_phenotype()
            pop1 = gp1.final_population
            print("DATASET:", ind.get_fitness(p1))

            gp2 = GP(
                problem=p2,
                representation=self.representation2,
                random_source=self.random_source,
                population_size=self.population2_size,
                initializer=InjectInitialPopulationWrapper([e.get_phenotype() for e in pop2], init),
                **self.kwargs2,
            )
            ind = gp2.evolve()
            self.bests.b2 = ind.get_phenotype()
            pop2 = gp2.final_population
            print("____________ Explanation:", ind.get_fitness(p2), ind.get_phenotype())

        return (self.bests.b1, self.bests.b2)
