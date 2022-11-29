
import glob
import numpy as np
import pandas as pd

from geneticengine.exceptions import GeneticEngineError

def load(folder_name: str, x_axis: str, y_axis: str):
    print(f"Loading from: {folder_name}")
    f_list = glob.glob(f"{folder_name}/*.csv")

    data = list()

    for f in f_list:
        df = pd.read_csv(f)
        try:
            df = df[[x_axis, y_axis]]
            data.append(df)
        except:
            continue
    
    if not data:
        raise GeneticEngineError(f"No files in folder {folder_name} have both columns ({x_axis} and {y_axis}). \n We recommend using the CSVCallback in the GP class to generate the csv files with the GP class parameter save_to_csv.")
    all_data = pd.concat(data, axis=0, ignore_index=True)
    
    return all_data

def load_w_extra(folder_name: str, x_axis: str, y_axis:str, extra_column: list):
    print(f"Loading from: {folder_name}")
    f_list = glob.glob(f"{folder_name}/*.csv")

    data = list()

    for f in f_list:
        df = pd.read_csv(f)
        try:
            df = df[[x_axis, y_axis] + extra_column]
            data.append(df)
        except:
            continue
    
    if not data:
        raise GeneticEngineError(f"No files in folder {folder_name} have both columns ({x_axis}, {y_axis} and {extra_column}). \n We recommend using the CSVCallback in the GP class to generate the csv files with the GP class parameter save_to_csv.")
    all_data = pd.concat(data, axis=0, ignore_index=True)
    
    return all_data

