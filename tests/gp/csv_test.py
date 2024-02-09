from abc import ABC
import numpy as np
import pandas as pd
from dataclasses import dataclass
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


class TestCSVCallback:
    def test_basic_fields(self, tmp_path):
        seed = 123
        max_generations = 5

        g = extract_grammar([Leaf], Root)

        path = tmp_path / "test.csv"

        # TODO: CSV callback
        objective = SingleObjectiveProblem(
            lambda p: 1,
            minimize=True,
        )

        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(g, max_depth=10),
            problem=objective,
            population_size=10,
            budget=EvaluationBudget(10 * max_generations),
            random=NativeRandomSource(seed),
        )
        gp.search()

        df = pd.read_csv(path)

        assert df.shape[0] == max_generations + 1
        assert df.shape[1] == 6 + objective.number_of_objectives()
        assert df.columns[0] == "Fitness Aggregated"
        assert all(v is not np.nan for v in df["Fitness Aggregated"])

        assert df.columns[1] == "Depth"
        assert all(v >= 0 for v in df["Depth"])

        assert df.columns[2] == "Nodes"
        assert all(v >= 0 for v in df["Nodes"])

        assert df.columns[3] == "Generations"
        assert all(v >= 0 for v in df["Generations"])

        assert df.columns[4] == "Execution Time"
        assert all(v >= 0 for v in df["Execution Time"])

        assert df.columns[5] == "Seed"
        assert all(v == seed for v in df["Seed"])

    def test_extra_fields(self, tmp_path):
        seed = 123
        max_generations = 5
        extra_val = 21323
        population_size = 10

        g = extract_grammar([Leaf], Root)

        path = tmp_path / "test.csv"

        # TODO: extra fields
        objective = SingleObjectiveProblem(
            lambda p: 1,
            minimize=True,
        )

        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(g, max_depth=10),
            problem=objective,
            population_size=population_size,
            budget=EvaluationBudget(population_size * max_generations),
            random=NativeRandomSource(seed),
        )
        gp.search()

        df = pd.read_csv(path)

        assert df.shape[0] == max_generations + 1
        assert df.shape[1] == 6 + objective.number_of_objectives() + 3
        assert all(v == extra_val for v in df["a"])
        assert all(v == g for v, g in zip(df["b"], df["Generations"]))
        assert all(v == population_size for v in df["c"])
