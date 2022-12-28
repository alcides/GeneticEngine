from __future__ import annotations

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import CrossoverOperator
from geneticengine.core.representations.api import Representation
from geneticengine.evaluators import Evaluator


class GenericCrossoverStep(GeneticStep):
    """Changes the population by crossing individuals two-by-two together,
    according to a given probability."""

    def __init__(
        self,
        probability: float,
        operator: CrossoverOperator | None = None,
    ):
        self.probability = probability
        self.operator = operator

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        assert len(population) == target_size
        self.operator = self.operator if self.operator else representation.get_crossover()

        mid = len(population) // 2
        retlist = []
        for (index, ind1, ind2) in zip(range(mid), population[:mid], population[mid:]):
            (n1, n2) = self.crossover(ind1, ind2, problem, representation, random_source, index, generation)
            retlist.append(n1)
            retlist.append(n2)

        # Fix odd-lengthed lists
        if len(population) % 2 != 0:
            retlist.append(population[-1])
        return retlist

    def crossover(
        self,
        individual1: Individual,
        individual2: Individual,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        index: int,
        generation: int,
    ):
        assert self.operator
        (g1, g2) = self.operator.crossover(
            individual1.genotype,
            individual2.genotype,
            problem,
            representation,
            random_source,
            index,
            generation,
        )
        return (
            Individual(g1, representation.genotype_to_phenotype),
            Individual(g2, representation.genotype_to_phenotype),
        )
