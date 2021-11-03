from typing import Callable, Tuple
from geneticengine.algorithms.gp.Individual import Individual
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.base import Representation
from geneticengine.core.representations.treebased import ProcessedGrammar


def create_cross_over(r: RandomSource, representation: Representation, pg: ProcessedGrammar, max_depth: int) -> Callable[[Individual,Individual],Tuple[Individual]]:
    
    def cross_over_double(individual1: Individual,individual2: Individual):
        (g1, g2) = representation.crossover_individuals(
                        r, pg, individual1.genotype, individual2.genotype, max_depth
                    )
        individual1 = Individual(g1)
        individual2 = Individual(g2)
        return (individual1,individual2)
        
    return cross_over_double