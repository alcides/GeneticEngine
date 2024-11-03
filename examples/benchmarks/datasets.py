from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_banknote(folder="examples/data") -> tuple[Any, Any, list[str]]:
    DATA_FILE_TRAIN = Path(folder) / "Banknote" / "Train.csv"

    # DATA_FILE_TEST = f"examples/data/{dataset_name}/Test.csv"

    bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter=" ")
    target = bunch.y
    data = bunch.drop(["y"], axis=1)

    target = bunch.y
    data = bunch.drop(["y"], axis=1)
    feature_names = list(data.columns.values)
    return data.values, target.values, feature_names


def get_vladislavleva(folder="examples/data") -> tuple[Any, Any, list[str]]:
    DATA_FILE_TRAIN = Path(folder) / "Vladislavleva4" / "Train.txt"

    bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter="\t")
    target = bunch.response
    data = bunch.drop(["response"], axis=1)

    feature_names = list(data.columns.values)
    return data.values, target.values, feature_names


def get_game_of_life(folder="examples/data") -> tuple[Any, Any]:
    DATA_FILE_TRAIN = Path(folder) / "GameOfLife" / "Train.csv"
    train = np.genfromtxt(DATA_FILE_TRAIN, skip_header=1, delimiter=",", dtype=int)
    X = train[:, :-1].reshape(train.shape[0], 3, 3)
    y = train[:, -1]
    return X, y
