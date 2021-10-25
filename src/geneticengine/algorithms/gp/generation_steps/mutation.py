from typing import Callable
from geneticengine.algorithms.gp.Individual import Individual
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.base import Representation
from geneticengine.core.representations.treebased import ProcessedGrammar


def create_mutation(r: RandomSource, representation: Representation, pg: ProcessedGrammar, probability_mutation: int) -> Callable[[Individual],Individual]:

    def mutation(individual: Individual):
        if r.randint(0, 100) < probability_mutation * 100:
            individual = Individual( genotype=representation.mutate_individual(
                                                r, pg, individual.genotype
                                            ),
                                        fitness=None,
                                    )
        return individual
        
    return mutation