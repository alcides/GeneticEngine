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

def plot_comparison(folder_names: list, labels: list, labels_name: str = 'Labels', x_axis: str = 'Generations', y_axis: str = 'Fitness', title: str = 'Fitness comparison', file_name = None):
    assert len(folder_names) == len(labels)
    
    all_data = list()
    
    for idx, folder_name in enumerate(folder_names):
        data = load(folder_name, x_axis, y_axis)
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
    
def plot_fitness_comparison(folder_names: list, labels: list, labels_name: str = 'Labels', x_axis: str = 'Generations', y_axis: str = 'Fitness', title: str = 'Fitness comparison', file_name = None):
    plot_comparison(folder_names=folder_names, labels=labels, labels_name=labels_name, x_axis=x_axis, y_axis=y_axis, title=title, file_name=file_name)
    
def plot_test_fitness_comparison(folder_names: list, labels: list, labels_name: str = 'Labels', x_axis: str = 'Generations', y_axis: str = 'Test fitness', title: str = 'Test fitness comparison', file_name = None):
    plot_comparison(folder_names=folder_names, labels=labels, labels_name=labels_name, x_axis=x_axis, y_axis=y_axis, title=title, file_name=file_name)
    
def plot_nodes_comparison(folder_names: list, labels: list, labels_name: str = 'Labels', x_axis: str = 'Generations', y_axis: str = 'Nodes', title: str = 'Nodes comparison', file_name = None):
    raise GeneticEngineError("Not yet implemented")
    # plot_comparison(folder_names=folder_names, labels=labels, labels_name=labels_name, x_axis=x_axis, y_axis=y_axis, title=title, file_name=file_name)
    

def plot_prods_comparison(folder_name: str, x_axis: str = 'Generations', extra: str = 'productions', y_axis: str = 'Fitness', title: str = 'Production comparison', file_name = None, take_out_prods: list = [ 'str', 'float', 'int' ], keep_in_prods: list = None):
    
    all_data = list()
    
    data = load_w_extra(folder_name, x_axis, y_axis, extra)
    prods = data[[extra]].values[0][0].split('<class \'')[1:]
    prods = list(map(lambda x: x.split('\'>:')[0], prods))
    def obtain_value(dictionary, prod):
        only_end = dictionary.split(prod+'\'>: ')[1]
        only_beginning = only_end.split(',')[0]
        try:
            return int(only_beginning)
        except:
            return int(only_beginning.split('}')[0])
    for prod in prods:
        prod = prod.split('.')[-1]
        if prod in take_out_prods:
            continue
        if keep_in_prods:
            if prod not in keep_in_prods:
                continue
        new_data = data.copy(deep=True)
        new_data['Occurences'] = data[['productions']].squeeze().map(lambda x: obtain_value(x, prod))
        new_data['Production'] = prod
        all_data.append(new_data[[x_axis, 'Occurences', 'Production']])
    
    df = pd.concat(all_data, axis=0, ignore_index=True)
    
    plt.close()
    sns.set_style("darkgrid")
    sns.set(font_scale=1.2)
    sns.set_style({"font.family": "serif"})

    a = sns.lineplot(
        data = df,
        x = x_axis,
        y = 'Occurences',
        hue = 'Production'
        )
    
    sns.set(font_scale=1.4)
    a.set_title(title)
    plt.tight_layout()
    
    if not file_name:
        file_name = title.replace(' ', '_') + '.pdf'
    plt.savefig(file_name)
    print(f"Saved figure to {file_name}.")
    
