from __future__ import annotations

import glob
from typing import Any
from typing import List

import matplotlib  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
import numpy as np
import pandas as pd
from palettable.colorbrewer.qualitative import Set2_7

from geneticengine.exceptions import GeneticEngineError

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def load(example_name, metric, single_value, aggregation, print_generations=False):
    search_folder = f"results/{example_name}/run_seed=*.csv"
    print(f"Loading from: {search_folder}")
    f_list = glob.glob(search_folder)

    data = list()

    for f in f_list:
        df = pd.read_csv(f)
        if not single_value:
            df = df[[metric, "number_of_the_generation"]]
            df = df.groupby("number_of_the_generation").agg(aggregation)
        data.append(df[metric].values)
        if print_generations:
            print(f, ": ", df[metric].values)
    return np.array(data)


def perc(data, size):
    median = np.zeros(size)
    perc_25 = np.zeros(size)
    perc_75 = np.zeros(size)
    for i in range(0, size):
        median[i] = np.median(data[:, i])
        perc_25[i] = np.percentile(data[:, i], 25)
        perc_75[i] = np.percentile(data[:, i], 75)
    return median, perc_25, perc_75


def plot_comparison(
    file_run_names,
    run_names,
    result_name="results/images/medians.pdf",
    metric="fitness",
    aggregation="mean",
    single_value=False,
):
    if len(file_run_names) != len(run_names) and len(run_names) != 0:
        raise GeneticEngineError(
            "The given [file_run_names] has a different length than the given [run_names]. Length should be same or keep enter an empty list for [run_names].",
        )
    if len(run_names) == 0:
        print("[run_names] is empty. Taking [file_run_names] as run_names.")
        run_names = file_run_names

    colors = Set2_7.mpl_colors
    plt.axes(frameon=0)
    plt.grid(axis="y", color="0.9", linestyle="-", linewidth=1)
    line_styles = ["solid", "dotted", "dashed", "dashdot"]

    for idx, file_run_name in enumerate(file_run_names):
        run_data = load(
            file_run_name,
            metric=metric,
            single_value=single_value,
            aggregation=aggregation,
        )
        try:
            n_generations = run_data.shape[1]
        except IndexError:
            run_data = load(
                file_run_name,
                metric=metric,
                single_value=single_value,
                aggregation=aggregation,
                print_generations=True,
            )
            raise GeneticEngineError(
                f"Index Error. \nMake sure the files you're loading in all have the same number of generations!",
            )
        x = np.arange(0, n_generations)

        med_run_data, perc_25_run_data, perc_75_run_data = perc(
            run_data,
            n_generations,
        )

        plt.plot(
            x,
            med_run_data,
            linewidth=2,
            color=colors[idx % len(colors)],
            linestyle=line_styles[idx % len(line_styles)],
        )

        plt.fill_between(
            x,
            perc_25_run_data,
            perc_75_run_data,
            alpha=0.25,
            linewidth=0,
            color=colors[idx % len(colors)],
            label="_nolegend_",
        )

    legend = plt.legend(run_names, loc=4)
    frame = legend.get_frame()
    frame.set_facecolor("1.0")
    frame.set_edgecolor("1.0")

    print(f"Saving image to path: {result_name}")
    plt.savefig(result_name)
    plt.close()


files: list[str] = [
    # 'Franklin/csvs/GoL/grammar_standard',
    # 'Franklin/csvs/GoL/grammar_row_col_cube',
    # 'Franklin/csvs/GoL/grammar_cube',
    # 'Franklin/csvs/GoL/grammar_row',
    # 'Franklin/csvs/GoL/grammar_col',
    # 'Franklin/csvs/GoL/grammar_row_col',
    # 'Franklin/csvs/GoL/grammar_sum_all'
]
run_names: list[str] = [
    # 'standard',
    # 'row col cube',
    # 'cube',
    # 'row',
    # 'col',
    # 'row col',
    # 'sum all'
]
files_noise = [
    "Franklin/csvs/GoL_noise/grammar_standard",
    "Franklin/csvs/GoL_noise/grammar_row_col_cube",
    "Franklin/csvs/GoL_noise/grammar_cube",
    "Franklin/csvs/GoL_noise/grammar_row",
    "Franklin/csvs/GoL_noise/grammar_col",
    "Franklin/csvs/GoL_noise/grammar_row_col",
    "Franklin/csvs/GoL_noise/grammar_sum_all",
]
run_names_noise = [
    "standard noise",
    "row col cube noise",
    "cube noise",
    "row noise",
    "col noise",
    "row col noise",
    "sum all noise",
]

# plot_comparison(
#     ['csvs/GoL/grammar_standard', 'csvs/GoL/grammar_standard(old)'],
#     ['new','old'],
#     # files + files_noise,
#     # run_names + run_names_noise,
#     result_name='results/images/compare_old_new.pdf',
#     # result_name='results/Franklin/images/accuracy/grammars_comparison_noise2.pdf'
# )
# plot_comparison(
#     ['csvs/GoL/grammar_standard', 'csvs/GoL/grammar_standard(old)'],
#     ['new','old'],
#     # files + files_noise,
#     # run_names + run_names_noise,
#     result_name='results/images/compare_time_old_new.pdf',
#     # result_name='results/Franklin/images/time/grammars_time_comparison_noise2.pdf',
#     metric='time_since_the_start_of_the_evolution'
# )
