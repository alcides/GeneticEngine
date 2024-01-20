from __future__ import annotations

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import CrossoverOperator
from geneticengine.core.representations.api import Representation
from geneticengine.core.evaluators import Evaluator


class GenericCrossoverStep(GeneticStep):
    """Changes the population by crossing individuals two-by-two together,
    according to a given probability."""

    def __init__(
        self,
        probability: float = 1,
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
        self.operator = self.operator if self.operator else representation.get_crossover()

        retlist: list[Individual] = []
        for i in range(target_size // 2):
            j = i % len(population)
            ind1, ind2 = population[j], population[j + 1]  # todo: select individuals using a selection method
            v = random_source.random_float(0, 1)
            if v <= self.probability:
                (n1, n2) = self.crossover(ind1, ind2, problem, representation, random_source, i, generation)
            else:
                (n1, n2) = (ind1, ind2)
            retlist.append(n1)
            retlist.append(n2)

        if len(retlist) < target_size:
            retlist.append(population[0])
        assert len(retlist) == target_size
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
