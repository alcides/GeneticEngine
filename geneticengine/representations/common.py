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
        def bound(i: int):
            interval = (representation.max_depth - representation.min_depth) + 1
            v = representation.min_depth + (i % interval)
            return v

        return [
            Individual(
                representation.instantiate(
                    r=random_source,
                    depth=bound(i),
                ),
                genotype_to_phenotype=representation.map,
            )
            for i in range(target_size)
        ]
