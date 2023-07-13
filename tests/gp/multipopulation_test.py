from dataclasses import dataclass
from geneticengine.algorithms.callbacks.callback import DebugCallback
from geneticengine.algorithms.gp.multipopulationgp import MultiPopulationGP
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation


@dataclass
class Option:
    a: int


def fitness_function(x: Option):
    return x.a


def test_multipopulation_basic():
    grammar = extract_grammar([Option], Option)

    representation = TreeBasedRepresentation(grammar=grammar, max_depth=2)
    problem1 = SingleObjectiveProblem(fitness_function=fitness_function, minimize=False)
    problem2 = SingleObjectiveProblem(fitness_function=fitness_function, minimize=True)
    problems = [problem1, problem2]
    r = RandomSource(seed=3)

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
    assert fs[0].get_fitness(problem1) > fs[1].get_fitness(problem2)
