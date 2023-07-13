from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


class GenericPopulationInitializer(PopulationInitializer):
    """Starts with an initial population, and relies on another initializer is
    necessary to fulfill the population size."""


    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        def bound(i: int):
            interval = (representation.max_depth - representation.min_depth) + 1
            v = representation.min_depth + (i % interval)
            return v

        return [
            Individual(
                representation.create_individual(
                    r=random_source,
                    depth=bound(i),
                ),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for i in range(target_size)
        ]
