from dataclasses import dataclass
from geneticengine.algorithms.callbacks.callback import DebugCallback
from geneticengine.algorithms.gp.multipopulationgp import MultiPopulationGP
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import Problem, SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


@dataclass
class Option:
    a: int


def fitness_function(x: Option):
    return x.a


grammar = extract_grammar([Option], Option)
representation = TreeBasedRepresentation(grammar=grammar, max_depth=2)
problem1: Problem = SingleObjectiveProblem(fitness_function=fitness_function, minimize=False)
problem2: Problem = SingleObjectiveProblem(fitness_function=fitness_function, minimize=True)
problems = [problem1, problem2]
r = NativeRandomSource(seed=3)

gp = MultiPopulationGP(
    representation=representation,
    random_source=r,
    problems=problems,
    population_sizes=[10, 10],
    stopping_criterium=GenerationStoppingCriterium(100),
    migration_size=2,
    callbacks=[DebugCallback()],
)
fs = gp.evolve()
print([(f.get_phenotype(), f.get_fitness(p).fitness_components[0]) for f, p in zip(fs, problems)])
