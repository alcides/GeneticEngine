from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.initializers import StandardInitializer

from geneticengine.grammar.grammar import Grammar

from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource, RandomSource
from geneticengine.representations.api import Representation
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.operators import InjectInitialPopulationWrapper
from geneticengine.representations.tree.treebased import TreeBasedRepresentation

a = TypeVar("a")
b = TypeVar("b")


class CooperativeGP(Generic[a, b]):
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
        random: Optional[RandomSource] = None,
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
            random (Source): The random number generator
            kwargs1 (dict): The extra arguments for the GP object of species1
            kwargs2 (dict): The extra arguments for the GP object of species2
        """
        self.g1 = grammar1
        self.g2 = grammar2
        self.ff = {"ff": function}

        self.population1_size = population1_size
        self.population2_size = population2_size
        self.coevolutions = coevolutions
        self.representation1 = representation1 or TreeBasedRepresentation(
            grammar=self.g1,
            decider=MaxDepthDecider(random, self.g1),
        )
        self.representation2 = representation2 or TreeBasedRepresentation(
            grammar=self.g2,
            decider=MaxDepthDecider(random, self.g2),
        )
        self.kwargs1 = kwargs1 or {}
        self.kwargs2 = kwargs2 or {}
        self.random = random or NativeRandomSource()

    def search(self) -> tuple[a, b]:
        @dataclass
        class Bests:
            b1: a
            b2: b

        b1: a = self.representation1.create_genotype(self.random)  # type: ignore
        b2: b = self.representation2.create_genotype(self.random)  # type: ignore
        self.bests = Bests(b1, b2)

        f = self.ff["ff"]
        init = StandardInitializer()

        def f1(x: a) -> float:
            return f(x, self.bests.b2)

        def f2(x: b) -> float:
            return f(self.bests.b1, x)

        p1 = SingleObjectiveProblem(fitness_function=f1, minimize=True)
        p2 = SingleObjectiveProblem(fitness_function=f2, minimize=False)

        pop1 = init.initialize(p1, self.representation1, self.random, self.population1_size)
        pop2 = init.initialize(p2, self.representation2, self.random, self.population1_size)

        for _ in range(self.coevolutions):
            # We create new problems to avoid results from previous iterations being cached.
            p1 = SingleObjectiveProblem(fitness_function=f1, minimize=True)
            p2 = SingleObjectiveProblem(fitness_function=f2, minimize=False)

            gp1 = GeneticProgramming(
                problem=p1,
                representation=self.representation1,
                random=self.random,
                population_size=self.population1_size,
                population_initializer=InjectInitialPopulationWrapper(
                    [e.get_phenotype() for e in pop1],
                    init,
                ),  # TODO: we might want to keep individuals, and not only the phenotypes.
                **self.kwargs1,
            )
            inds = gp1.search()
            self.bests.b1 = inds[0].get_phenotype()

            gp2 = GeneticProgramming(
                problem=p2,
                representation=self.representation2,
                random=self.random,
                population_size=self.population2_size,
                population_initializer=InjectInitialPopulationWrapper([e.get_phenotype() for e in pop2], init),
                **self.kwargs2,
            )
            inds = gp2.search()
            self.bests.b2 = inds[0].get_phenotype()

        return (self.bests.b1, self.bests.b2)
