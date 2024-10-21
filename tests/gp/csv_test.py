from abc import ABC
import pandas as pd
from dataclasses import dataclass
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.evaluation.recorder import CSVSearchRecorder
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    v: int


def is_sorted(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1)) or all(
        lst[i] >= lst[i + 1] for i in range(len(lst) - 1)
    )


class TestCSVCallback:
    def test_basic_fields(self, tmp_path):
        seed = 123
        max_generations = 10
        population_size = 10

        g = extract_grammar([Leaf], Root)

        path = tmp_path / "test.csv"

        objective = SingleObjectiveProblem(
            lambda p: abs(p.v - 2024),
            minimize=True,
        )
        r = NativeRandomSource(seed)
        decider = MaxDepthDecider(r, g, max_depth=10)
        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(g, decider=decider),
            problem=objective,
            population_size=population_size,
            budget=EvaluationBudget(population_size * max_generations),
            random=r,
            tracker=ProgressTracker(
                objective,
                recorders=[CSVSearchRecorder(csv_path=path, problem=objective)],
            ),
        )
        gp.search()
        df = pd.read_csv(path)

        assert list(df.columns) == ["Execution Time", "Phenotype", "Fitness0"]
        assert is_sorted(df["Fitness0"].values)
        assert is_sorted(df["Execution Time"].values)

    def test_extra_fields(self, tmp_path):
        seed = 123
        max_generations = 10
        population_size = 10

        g = extract_grammar([Leaf], Root)

        path = tmp_path / "test.csv"

        objective = SingleObjectiveProblem(
            lambda p: abs(p.v - 2024),
            minimize=True,
        )
        r = NativeRandomSource(seed)
        decider = MaxDepthDecider(r, g, max_depth=10)
        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(g, decider=decider),
            problem=objective,
            population_size=population_size,
            budget=EvaluationBudget(population_size * max_generations),
            random=NativeRandomSource(seed),
            tracker=ProgressTracker(
                objective,
                recorders=[
                    CSVSearchRecorder(csv_path=path, problem=objective, extra_fields={"Seed": lambda t, i, p: seed}),
                ],
            ),
        )
        gp.search()
        df = pd.read_csv(path)

        assert list(df.columns) == ["Execution Time", "Phenotype", "Fitness0", "Seed"]
        assert is_sorted(df["Fitness0"].values)
        assert is_sorted(df["Execution Time"].values)
        assert all([x == seed for x in df["Seed"].values])

    def test_fields(self, tmp_path):
        seed = 123
        max_generations = 10
        population_size = 10

        g = extract_grammar([Leaf], Root)

        path = tmp_path / "test.csv"

        objective = SingleObjectiveProblem(
            lambda p: abs(p.v - 2024),
            minimize=True,
        )

        r = NativeRandomSource(seed)
        decider = MaxDepthDecider(r, g, max_depth=10)
        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(g, decider=decider),
            problem=objective,
            population_size=population_size,
            budget=EvaluationBudget(population_size * max_generations),
            random=NativeRandomSource(seed),
            tracker=ProgressTracker(
                objective,
                recorders=[CSVSearchRecorder(csv_path=path, problem=objective, fields={"Seed": lambda t, i, p: seed})],
            ),
        )
        gp.search()
        df = pd.read_csv(path)

        assert list(df.columns) == ["Seed"]
        assert all([x == seed for x in df["Seed"].values])
