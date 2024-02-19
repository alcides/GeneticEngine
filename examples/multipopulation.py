from dataclasses import dataclass
from geneticengine.algorithms.gp.multipopulationgp import MultiPopulationGP
from geneticengine.evaluation.budget import EvaluationBudget

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
    problem=problem1,
    budget=EvaluationBudget(100),
    representation=representation,
    random=r,
    population_sizes=[10, 10],
    migration_size=2,
)
f = gp.search()
print((f.get_phenotype(), f.get_fitness(problem1).fitness_components[0]))
