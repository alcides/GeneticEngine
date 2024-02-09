from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import SolutionRepresentation


class GenericPopulationInitializer(PopulationInitializer):
    """Initializes any representation."""

    def initialize(
        self,
        problem: Problem,
        representation: SolutionRepresentation,
        random_source: RandomSource,
        target_size: int,
    ) -> list[Individual]:
        return [
            Individual(
                representation.instantiate(
                    random=random_source,
                ),
                genotype_to_phenotype=representation.map,
            )
            for i in range(target_size)
        ]
