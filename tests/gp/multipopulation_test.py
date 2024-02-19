from dataclasses import dataclass
import sys
from geneticengine.algorithms.gp.multipopulationgp import MultiPopulationGP
from geneticengine.evaluation.budget import EvaluationBudget

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


@dataclass
class Option:
    a: int


def fitness_function(x: Option):
    return x.a


def test_multipopulation_basic():
    grammar = extract_grammar([Option], Option)

    representation = TreeBasedRepresentation(grammar=grammar, max_depth=2)
    problem1 = SingleObjectiveProblem(fitness_function=fitness_function, minimize=False)
    # problem2 = SingleObjectiveProblem(fitness_function=fitness_function, minimize=True)
    # problems = [problem1, problem2]
    r = NativeRandomSource(seed=3)

    gp = MultiPopulationGP(
        representation=representation,
        random=r,
        problem=problem1,
        population_sizes=[10, 10],
        budget=EvaluationBudget(20 * 100),
        migration_size=2,
    )

    ind = gp.search()
    min_size = -(sys.maxsize - 1)
    assert ind.get_phenotype().a > min_size
