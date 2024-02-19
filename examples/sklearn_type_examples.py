from __future__ import annotations

import pandas as pd

from geml.classifiers import GeneticProgrammingClassifier
from geml.classifiers import HillClimbingClassifier
from geml.regressors import GeneticProgrammingRegressor
from geml.regressors import HillClimbingRegressor

# Examples of how to use the off_the_shelf classifiers and regressors.
# ===================================

# ===================================
# Classifiers
# ===================================

DATASET_NAME = "Banknote"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.csv"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.csv"

bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter=" ")
target = bunch.y
data = bunch.drop(["y"], axis=1)

print("GP Classifier")
model = GeneticProgrammingClassifier()
model.fit(data, target)
print(model.predict(data.iloc[0:5]))
print(target.iloc[0:5].values)

print("HC Classifier")
model = HillClimbingClassifier()
model.fit(data, target)
print(model.predict(data.iloc[0:5]))
print(target.iloc[0:5].values)


# ===================================
# Regressors
# ===================================

DATASET_NAME = "Vladislavleva4"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.txt"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.txt"

bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter="\t")
target = bunch.response
data = bunch.drop(["response"], axis=1)

print("GP Regressor")
model = GeneticProgrammingRegressor(metric="r2")
model.fit(data, target)
print(model.predict(data.iloc[0:5]))
print(target.iloc[0:5].values)

print("HC Regressor")
model = HillClimbingRegressor()
model.fit(data, target)
print(model.predict(data.iloc[0:5]))
print(target.iloc[0:5].values)
