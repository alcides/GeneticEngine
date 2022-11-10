from __future__ import annotations

import glob
from typing import Any
from typing import List

import matplotlib  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import numpy as np
import pandas as pd
import seaborn as sns

from geneticengine.exceptions import GeneticEngineError
from geneticengine.visualization.utils import *

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def plot_fitness_comparison(folder_names: list, labels: list, minimize: bool, labels_name: str = 'Labels', x_axis: str = 'number_of_the_generation', y_axis: str = 'fitness', title: str = 'Fitness comparison', file_name = None):
    
    all_data = list()
    
    for idx, folder_name in enumerate(folder_names):
        data = load(folder_name, x_axis, y_axis, minimize)
        data[labels_name] = labels[idx]
        all_data.append(data)

    all_data = pd.concat(all_data, axis=0, ignore_index=True)

    plt.close()
    sns.set_style("darkgrid")
    sns.set(font_scale=1.2)
    sns.set_style({"font.family": "serif"})

    a = sns.lineplot(
        data=all_data,
        x = x_axis,
        y = y_axis,
        hue = labels_name
        )
    
    sns.set(font_scale=1.4)
    a.set_title(title)
    plt.tight_layout()
    
    if not file_name:
        file_name = title.replace(' ', '_') + '.pdf'
    plt.savefig(file_name)
    print(f"Saved figure to {file_name}.")
    