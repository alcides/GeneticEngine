from __future__ import annotations
import random
import sys

from geneticengine.grammars.dynamic_grammar import edit_distance
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.dynamic_grammar import create_grammar_nodes
from geneticengine.core.random.sources import RandomSource

from geneticengine.algorithms.callbacks.csv_callback import CSVCallback


def create_dynamic_grammar(grammar_seed, n_class_abc, n_class_0_children, n_class_2_children, max_var_per_class):
    
    (list, starting_node) = create_grammar_nodes(
        grammar_seed,
        n_class_abc,
        n_class_0_children,
        n_class_2_children,
        max_var_per_class,
    )

    return extract_grammar(list, starting_node)

def create_target_individual(grammar_seed, g):
    
    r = RandomSource(grammar_seed)
    representation = treebased_representation
    target_individual = representation.create_individual(r, g, depth=10)
    individual_phenotype = representation.genotype_to_phenotype(
        g, target_individual)
    return individual_phenotype

    
if __name__ == "__main__":
    # run $ python3 dynamic_grammar_example.py <number_abc_class> <number_terminal_class> <number_nterminal_class> <max_vars_per_class> <number_of_runs>
    args = sys.argv[1:]
    assert len(args) == 5, 'Incorrect input!! It should be e.g: "$ python3 dynamic_grammar_example.py 2 3 3 5 4"'
    seeds_used = [1123]
    for i in range((int(args[-1]))):
        #grammar_seed = 100 + i
        
        #do while
        #generate different random grammar seed each iteration
        while True:
            grammar_seed = random.randint(1, 100+i)
            if grammar_seed not in seeds_used:
                seeds_used.append(grammar_seed)
                break
        
        
        g = create_dynamic_grammar(grammar_seed, int(args[0]), int(args[1]), int(args[2]), int(args[3]))
        print(g)
        
        target_individual= create_target_individual(grammar_seed, g)
        
        def fitness_function(n):
            return edit_distance(str(n), str(target_individual))
        
        def evolve(g, seed, mode):
            alg = GP(
                g,
                representation=treebased_representation,
                problem=SingleObjectiveProblem(
                    fitness_function=fitness_function,
                    minimize=True,
                    target_fitness=0,
                ),
                population_size=100,
                number_of_generations=25,
                timer_stop_criteria=mode,
                seed=seed,
                save_to_csv=CSVCallback(
                    filename=("dynamic_grammar_"+str(grammar_seed))),
            )
            (b, bf, bp) = alg.evolve()
            return b, bf

        bf, b = evolve(g, 1123, False)
        print(b)
        print(f"With fitness: {bf}")
