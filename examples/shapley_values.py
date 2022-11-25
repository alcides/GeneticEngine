from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from geneticengine.off_the_shelf.regressors import GeneticProgrammingRegressor

X, y = shap.datasets.boston()

print("GP Regressor")
model = GeneticProgrammingRegressor(metric="r2")
model.fit(X, y, verbose=1)

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


r = permutation_importance(model, X, y, n_repeats=30, random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(
            f"{X.columns[i]:<8}"
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}",
        )
