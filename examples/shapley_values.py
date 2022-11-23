from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import shap

from geneticengine.off_the_shelf.regressors import GeneticProgrammingRegressor

X, y = shap.datasets.boston()

print("GP Regressor")
model = GeneticProgrammingRegressor(metric="r2")
model.fit(X, y)

X_train_summary = shap.kmeans(X, 10)
explainer = shap.KernelExplainer(model.predict, X_train_summary)
shap_values = explainer.shap_values(X)


fig = shap.summary_plot(shap_values, X.columns, show=False)
plt.savefig("scratch1.png")

plt.figure()
shap.decision_plot(
    explainer.expected_value,
    shap_values,
    X.columns,
    show=False,
    link="logit",
    feature_order="hclust",
)
plt.savefig("scratch2.png")
