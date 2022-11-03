from __future__ import annotations

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.initializations import full_method
from geneticengine.core.representations.tree.initializations import pi_grow_method
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation


class RampedHalfAndHalfInitializer(PopulationInitializer):
    """Half of the individuals are created with the maximum depth, and the
    other half with different values of maximum depth between the minimum and
    the maximum.

    There's an equal chance of using full or grow method.
    """

    def initialize(
        self,
        p: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        assert isinstance(representation, TreeBasedRepresentation)
        return [
            Individual(
                representation.create_individual(
                    random_source,
                    random_source.randint(
                        representation.min_depth,
                        representation.max_depth,
                    ),
                    method=random_source.choice([pi_grow_method, full_method]),
                ),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for _ in range(target_size)
        ]
