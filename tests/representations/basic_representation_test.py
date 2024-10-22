import copy
from typing import Any
from geneticengine.algorithms.hill_climbing import HC
from geneticengine.algorithms.one_plus_one import OnePlusOne
from geneticengine.algorithms.random_search import RandomSearch
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import RepresentationWithMutation, Representation

MAX_NUMBER = 200
MAX_ELEMENTS = 10


class LinearRepresentation(Representation[list[int], int], RepresentationWithMutation[list[int]]):
    def create_genotype(self, random: RandomSource, **kwargs) -> list[int]:
        return [random.randint(0, MAX_NUMBER) for _ in range(MAX_ELEMENTS)]

    def genotype_to_phenotype(self, internal: list[int]) -> int:
        return sum(internal)

    def mutate(self, random: RandomSource, genotype: list[int], **kwargs: Any) -> list[int]:
        nc = copy.deepcopy(genotype)
        ind = random.randint(0, len(nc) - 1)
        nc[ind] = random.randint(0, MAX_NUMBER)
        return nc


class TestBasicRepresentation:
    def test_random_search(self):
        p = SingleObjectiveProblem(fitness_function=lambda x: abs(2024 - x), minimize=True)
        rs = RandomSearch(problem=p, budget=EvaluationBudget(100), representation=LinearRepresentation())
        v = rs.search()[0]
        assert isinstance(v.get_phenotype(), int)
        assert 0 <= v.get_phenotype() <= MAX_NUMBER * MAX_ELEMENTS

    def test_one_plus_one(self):
        p = SingleObjectiveProblem(fitness_function=lambda x: abs(2024 - x), minimize=True)
        rs = OnePlusOne(problem=p, budget=EvaluationBudget(100), representation=LinearRepresentation())
        v = rs.search()[0]
        assert isinstance(v.get_phenotype(), int)
        assert 0 <= v.get_phenotype() <= MAX_NUMBER * MAX_ELEMENTS

    def test_hill_climbing(self):
        p = SingleObjectiveProblem(fitness_function=lambda x: abs(2024 - x), minimize=True)
        rs = HC(problem=p, budget=EvaluationBudget(100), representation=LinearRepresentation())
        v = rs.search()[0]
        assert isinstance(v.get_phenotype(), int)
        assert 0 <= v.get_phenotype() <= MAX_NUMBER * MAX_ELEMENTS
