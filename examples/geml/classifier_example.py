import pandas as pd
from sklearn.metrics import f1_score
from geml.classifiers import (
    GeneticProgrammingClassifier,
    HillClimbingClassifier,
    OnePlusOneClassifier,
    RandomSearchClassifier,
)


DATASET_NAME = "Banknote"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.csv"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.csv"


bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter=" ", header=0)
target = bunch.y
data = bunch.drop(["y"], axis=1)
test = pd.read_csv(DATA_FILE_TEST, delimiter=" ", header=0)
test_data = test.drop(["y"], axis=1)
test_target = test.y

seed = 1337


for model_class in [GeneticProgrammingClassifier, HillClimbingClassifier, RandomSearchClassifier, OnePlusOneClassifier]:
    model = model_class(max_time=20.0, seed=seed)
    model.fit(data, target)
    y_pred = model.predict(test_data)
    r2 = f1_score(test_target, y_pred)
    print(f"{model}: {r2}")
