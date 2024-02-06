from geneticengine.algorithms.random_search import RandomSearch
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import SolutionRepresentation

MAX_NUMBER = 200
MAX_ELEMENTS = 10


class LinearRepresentation(SolutionRepresentation[list[int], int]):
    def instantiate(self, random: RandomSource, **kwargs) -> list[int]:
        return [random.randint(0, MAX_NUMBER) for _ in range(MAX_ELEMENTS)]

    def map(self, internal: list[int]) -> int:
        return sum(internal)


class TestBasicRepresentation:
    def test_basic_representation(self):
        p = SingleObjectiveProblem(fitness_function=lambda x: abs(2024 - x), minimize=True)
        rs = RandomSearch(problem=p, budget=EvaluationBudget(100), representation=LinearRepresentation())
        v = rs.search()
        assert isinstance(v.get_phenotype(), int)
        assert 0 <= v.get_phenotype() <= MAX_NUMBER * MAX_ELEMENTS
