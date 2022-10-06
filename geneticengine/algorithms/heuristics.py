from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from operator import attrgetter

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.problems import FitnessType
from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem


class Heuristics(ABC):
    @abstractmethod
    def evolve(self, verbose):
        pass

    # TODO: test this function
    def get_best_individual(
        self,
        p: Problem,
        individuals: list[Individual],
    ) -> Individual:
        best_individual: Individual
        if isinstance(p, SingleObjectiveProblem):
            fitnesses = [self.evaluate(x) for x in individuals]
            assert all(isinstance(x, float) for x in fitnesses)
            if p.minimize:
                best_individual = min(individuals, key=attrgetter("fitness"))
            else:
                best_individual = max(individuals, key=attrgetter("fitness"))

        elif isinstance(p, MultiObjectiveProblem):
            fitnesses = [self.evaluate(x) for x in individuals]
            assert all(isinstance(x, list) for x in fitnesses)

            def single_criteria(i: Individual) -> float:
                return sum((m and -f or f) for (f, m) in zip(i.fitness, p.minimize))

            return max(individuals, key=single_criteria)

        return best_individual

    def keyfitness(self):
        if self.problem.minimize:
            return lambda x: self.evaluate(x)
        else:
            return lambda x: -self.evaluate(x)

    def evaluate(self, individual: Individual) -> FitnessType:
        if individual.fitness is None:
            phenotype = self.representation.genotype_to_phenotype(
                self.grammar,
                individual.genotype,
            )
            individual.fitness = self.problem.evaluate(phenotype)
        return individual.fitness

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
