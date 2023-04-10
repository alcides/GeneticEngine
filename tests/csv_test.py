from abc import ABC
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation


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

        csv_callback = CSVCallback(path)
        objective = SingleObjectiveProblem(
            lambda p: 1,
            minimize=True,
        )

        gp = GP(
            representation=TreeBasedRepresentation(g, max_depth=10),
            problem=objective,
            population_size=10,
            stopping_criterium=GenerationStoppingCriterium(max_generations=max_generations),
            callbacks=[csv_callback],
            random_source=RandomSource(seed),
        )
        gp.evolve()

        df = pd.read_csv(path)

        assert df.shape[0] == max_generations
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

        os.remove(path)
