from typing import Iterator
from geneticengine.solutions.individual import PhenotypicIndividual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation


class GenericPopulationInitializer(PopulationInitializer):
    """Initializes any representation."""

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random: RandomSource,
        target_size: int,
    ) -> Iterator[PhenotypicIndividual]:
        for i in range(target_size):
            yield PhenotypicIndividual(
                representation.create_genotype(
                    random=random,
                ),
                representation=representation,
            )
