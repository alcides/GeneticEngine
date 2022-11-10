
import glob
import numpy as np
import pandas as pd


def load(folder_name: str, x_axis: str, y_axis: str, minimize: bool):
    print(f"Loading from: {folder_name}")
    f_list = glob.glob(f"{folder_name}/*.csv")

    data = list()

    for f in f_list:
        df = pd.read_csv(f)
        df = df[[x_axis, y_axis]]
        data.append(df)
        
    all_data = pd.concat(data, axis=0, ignore_index=True)

    if minimize:
        averaged_data = all_data.groupby([x_axis], as_index=False).max()
    else:
        averaged_data = all_data.groupby([x_axis], as_index=False).min()
    
    return averaged_data

