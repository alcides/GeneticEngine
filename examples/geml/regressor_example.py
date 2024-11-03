import pandas as pd
from sklearn.metrics import r2_score
from geml.regressors import (
    GeneticProgrammingRegressor,
    HillClimbingRegressor,
    OnePlusOneRegressor,
    RandomSearchRegressor,
)
from geml.regressors import model as extract_model


DATASET_NAME = "Vladislavleva4"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.txt"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.txt"

bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter="\t")
target = bunch.response
data = bunch.drop(["response"], axis=1)
test = pd.read_csv(DATA_FILE_TEST, delimiter="\t")
test_data = test.drop(["response"], axis=1)
test_target = test.response

seed = 1337


for model_class in [GeneticProgrammingRegressor, HillClimbingRegressor, RandomSearchRegressor, OnePlusOneRegressor]:
    model = model_class(max_time=20.0, seed=seed)
    model.fit(data, target)
    y_pred = model.predict(test_data)
    r2 = r2_score(test_target, y_pred)
    print(f"{model} ({extract_model(model)}): {r2}")
