from __future__ import annotations

import random as r

import pandas as pd


def introduce_noise(file_name, label_col="label", percentage_noise=10):
    if not isinstance(percentage_noise, int) or percentage_noise <= 0 or percentage_noise > 100:
        raise TypeError(
            "[percentage_noise] should be an int between 0 and 100.",
        )
    data = pd.read_csv(f"examples/data/{file_name}.csv")

    label_data = data[label_col]
    for idx, _ in enumerate(label_data):
        if r.randint(1, 100) <= percentage_noise:
            if label_data[idx] == 1:
                label_data[idx] = 0
            else:
                label_data[idx] = 1

    data[label_col] = label_data
    data.to_csv(f"examples/data/{file_name}_noise.csv", index=False)


introduce_noise("GameOfLife_noise/Train")
