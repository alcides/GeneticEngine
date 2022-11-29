from __future__ import annotations

import glob
from typing import Any
from typing import List

import matplotlib  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import numpy as np
import pandas as pd
import seaborn as sns # type: ignore

from geneticengine.exceptions import GeneticEngineError
from geneticengine.visualization.utils import *

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def plot_comparison(folder_names: list, labels: list, labels_name: str = 'Labels', x_axis: str = 'Generations', y_axis: str = 'Fitness', title: str = 'Fitness comparison', file_name = None):
    '''
        Plots a figure with lines for each folder (average with shades for std) with each folder from folder_names named with the corresponding labels_name.
    '''
    assert len(folder_names) == len(labels)
    
    all_data = list()
    
    for idx, folder_name in enumerate(folder_names):
        data = load(folder_name, x_axis, y_axis)
        data[labels_name] = labels[idx]
        all_data.append(data)

    all_data = pd.concat(all_data, axis=0, ignore_index=True)

    # --------------------
    palette = dict()
    for idx, label in enumerate(labels):
        palette[label] = f'C{idx}'
        
    plt.close()
    sns.set_style("darkgrid")
    sns.set(font_scale=1.2)
    sns.set_style({"font.family": "serif"})

    a = sns.lineplot(
        data=all_data,
        x = x_axis,
        y = y_axis,
        hue = labels_name,
        palette=palette,
        )
    
    sns.set(font_scale=1.4)
    a.set_title(title)
    plt.tight_layout()
    
    if not file_name:
        file_name = title.replace(' ', '_') + '.pdf'
    plt.savefig(file_name)
    print(f"Saved figure to {file_name}.")
    
def plot_fitness_comparison(folder_names: list, labels: list, labels_name: str = 'Labels', x_axis: str = 'Generations', y_axis: str = 'Fitness', title: str = 'Fitness comparison', file_name = None):
    '''
        Plots a figure with lines for each folder (average with shades for std) with each folder from folder_names named with the corresponding labels_name.
        In this case, the fitness is plotted
    '''
    plot_comparison(folder_names=folder_names, labels=labels, labels_name=labels_name, x_axis=x_axis, y_axis=y_axis, title=title, file_name=file_name)
    
def plot_test_fitness_comparison(folder_names: list, labels: list, labels_name: str = 'Labels', x_axis: str = 'Generations', y_axis: str = 'Test fitness', title: str = 'Test fitness comparison', file_name = None):
    '''
        Plots a figure with lines for each folder (average with shades for std) with each folder from folder_names named with the corresponding labels_name.
        In this case, the test fitness is plotted
    '''
    plot_comparison(folder_names=folder_names, labels=labels, labels_name=labels_name, x_axis=x_axis, y_axis=y_axis, title=title, file_name=file_name)
    
def plot_nodes_comparison(folder_names: list, labels: list, labels_name: str = 'Labels', x_axis: str = 'Generations', y_axis: str = 'Nodes', title: str = 'Nodes comparison', file_name = None):
    '''
        Plots a figure with lines for each folder (average with shades for std) with each folder from folder_names named with the corresponding labels_name.
        In this case, the nodes are plotted
    '''
    plot_comparison(folder_names=folder_names, labels=labels, labels_name=labels_name, x_axis=x_axis, y_axis=y_axis, title=title, file_name=file_name)

def plot_prods_comparison(folder_name: str, x_axis: str = 'Generations', extra: str = 'productions', y_axis: str = 'Fitness', title: str = 'Production comparison', file_name = None, take_out_prods: list = [ 'str', 'float', 'int' ], keep_in_prods: list | None = None, normalized_on_nodes: bool = False):
    '''
        Plots a figure with lines for each production (average with shades for std) in the grammar (you can use take_out_prods and keep_in_prods to take out and keep in prods). Only a single folder can be given.
    '''
    
    all_data = list()
    
    extra_cols = [extra]
    if normalized_on_nodes:
        extra_cols = [extra, 'Nodes']
    data = load_w_extra(folder_name, x_axis, y_axis, extra_cols)
    prods = data[[extra]].values[0][0].split('<class \'')[1:]
    prods = list(map(lambda x: x.split('\'>:')[0], prods))
    prods = list(map(lambda x: x.split('.')[-1], prods))
    if keep_in_prods:
        prods = [ prod for prod in prods if (prod in keep_in_prods) and (prod not in take_out_prods) ]
    else:
        prods = [ prod for prod in prods if (prod not in take_out_prods) ]
    def obtain_value(dictionary, prod):
        only_end = dictionary.split(prod+'\'>: ')[1]
        only_beginning = only_end.split(',')[0]
        try:
            return int(only_beginning)
        except:
            return int(only_beginning.split('}')[0])
    for prod in prods:
        new_data = data.copy(deep=True)
        new_data['Occurences'] = data[['productions']].squeeze().map(lambda x: obtain_value(x, prod))
        if normalized_on_nodes:
            new_data['Occurences'] = new_data['Occurences'] / new_data['Nodes']
        new_data['Production'] = prod
        all_data.append(new_data[[x_axis, 'Occurences', 'Production']])
    
    df = pd.concat(all_data, axis=0, ignore_index=True)

    # --------------------
    palette = dict()
    for idx, prod in enumerate(prods):
        palette[prod] = f'C{idx}'
    
    plt.close()
    sns.set_style("darkgrid")
    sns.set(font_scale=1.2)
    sns.set_style({"font.family": "serif"})

    a = sns.lineplot(
        data = df,
        x = x_axis,
        y = 'Occurences',
        hue = 'Production',
        palette=palette,
        )
    
    sns.set(font_scale=1.4)
    a.set_title(title)
    plt.tight_layout()
    
    if not file_name:
        file_name = title.replace(' ', '_') + '.pdf'
    plt.savefig(file_name)
    print(f"Saved figure to {file_name}.")
    
