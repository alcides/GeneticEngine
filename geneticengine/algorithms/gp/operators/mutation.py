from __future__ import annotations

from typing import Any
from typing import TypeVar

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.core.fitness_helpers import best_individual
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import MutationOperator
from geneticengine.core.representations.api import Representation
from geneticengine.core.evaluators import Evaluator


class GenericMutationStep(GeneticStep):
    """Applies a mutation to individuals with a given probability."""

    def __init__(
        self,
        probability: float = 1,
        operator: MutationOperator | None = None,
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
        if not self.operator:
            self.operator = representation.get_mutation()

        ret = []
        for index, ind in enumerate(population[:target_size]):
            v = random_source.random_float(0, 1)
            if v <= self.probability:
                print("beofre", ind.genotype)
                mutated = self.operator.mutate(
                    ind.genotype,
                    problem,
                    evaluator,
                    representation,
                    random_source,
                    index,
                    generation,
                )
                print("after", mutated)
                nind = self.wrap(representation, mutated)
                ret.append(nind)
            else:
                ret.append(ind)

        return ret

    def wrap(self, representation: Representation, genotype: Any) -> Individual:
        return Individual(
            genotype=genotype,
            genotype_to_phenotype=representation.genotype_to_phenotype,
        )


g = TypeVar("g")


class HillClimbingMutation(MutationOperator[g]):
    def __init__(self, n_candidates: int = 5, basic_mutator: MutationOperator | None = None):
        self.n_candidates = n_candidates
        self.basic_mutator = basic_mutator

    def mutate(
        self,
        genotype: g,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> g:
        basic_mutator = self.basic_mutator if self.basic_mutator else representation.get_mutation()

        new_genotypes = [genotype] + [
            basic_mutator.mutate(
                genotype,
                problem,
                evaluator,
                representation,
                random_source,
                index_in_population,
                generation,
            )
            for _ in range(self.n_candidates)
        ]
        new_individuals = [
            Individual(genotype=g, genotype_to_phenotype=representation.genotype_to_phenotype) for g in new_genotypes
        ]

        evaluator.eval(problem, new_individuals)
        bi = best_individual(new_individuals, problem)

        return bi.genotype
