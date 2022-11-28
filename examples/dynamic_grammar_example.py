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

(list, starting_node) = create_grammar_nodes(
    123,
    n_class_abc=2,
    n_class_0_children=3,
    n_class_2_children=3,
)

print(list)
print()
g = extract_grammar(list, starting_node)
print(g)
 

# TODO: nodes with the same name length


#target_individual = generateIndividual(depth= 10)

def fitness_function(n):
    
    #return edit_distance(str(n), str(target_individual))
    return 0

def evolve(g, seed, mode):
    alg = GP(
        g,
        representation=treebased_representation,
        problem=SingleObjectiveProblem(
            fitness_function=fitness_function,
            minimize=True,
            target_fitness=0,
        ),
        population_size=20,
        number_of_generations=5,
        timer_stop_criteria=mode,
        seed=seed,
    )
    (b, bf, bp) = alg.evolve()
    return b, bf


if __name__ == "__main__":
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
