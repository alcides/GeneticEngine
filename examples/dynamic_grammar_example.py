from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated
from geneticengine.grammars.dynamic_grammar import edit_distance
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.problems import SingleObjectiveProblem


from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.dynamic_grammar import create_grammar_nodes
from geneticengine.metahandlers.ints import IntRange

from geneticengine.core.random.sources import RandomSource

seed = 123
(list, starting_node) = create_grammar_nodes(
    seed,
    n_class_abc=2,
    n_class_0_children=3,
    n_class_2_children=3,
)

print(list)
print()
g = extract_grammar(list, starting_node)
print(g)

r = RandomSource(seed)

representation = treebased_representation
target_individual = representation.create_individual(r, g, depth= 10)
individual_phenotype = representation.genotype_to_phenotype(g, target_individual)


def fitness_function(n):
    return edit_distance(str(n), str(individual_phenotype))
    #return 0

def evolve(g, seed, mode):
    alg = GP(
        g,
        representation=treebased_representation,
        problem=SingleObjectiveProblem(
            fitness_function=fitness_function,
            minimize=True,
            target_fitness=0,
        ),
        population_size=50,
        number_of_generations=50,
        timer_stop_criteria=mode,
        seed=seed,
    )
    (b, bf, bp) = alg.evolve()
    return b, bf


if __name__ == "__main__":
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
