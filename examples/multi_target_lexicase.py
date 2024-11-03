from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import Annotated

import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils._param_validation import InvalidParameterError

from geml.simplegp import SimpleGP
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geml.grammars.basic_math import Exp
from geml.grammars.basic_math import SafeDiv
from geml.grammars.basic_math import SafeLog
from geml.grammars.basic_math import SafeSqrt
from geml.grammars.basic_math import Sin
from geml.grammars.basic_math import Tanh
from geml.grammars.sgp import Minus
from geml.grammars.sgp import Mul
from geml.grammars.sgp import Number
from geml.grammars.sgp import Plus
from geml.grammars.sgp import Var
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.grammar.metahandlers.vars import VarRange

# ===================================
# This is an example of a Multi target regression problem using normal GP,
# with a lexicase selection algorithm as the parent selection.
# We used the Linnerud dataset from sklearn library
# ===================================

BAD_FITNESS = 10000000

# Load the data from Sklearn
bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
X, y = bunch["data"], bunch["target"]


targets = y.values.tolist()

feature_names = bunch["feature_names"]
target_names = bunch["target_names"]

feature_indices = {}
for i, n in enumerate(feature_names):
    feature_indices[n] = i


Var.__init__.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
Var.feature_indices = feature_indices  # type: ignore


@dataclass
class Literal(Number):
    val: Annotated[int, IntRange(0, 9)]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


@dataclass
class NumberList:
    lst: Annotated[list[Number], ListSizeBetween(3, 3)]

    def evaluate(self, **kwargs):
        return [n.evaluate(**kwargs) for n in self.lst]

    def __str__(self) -> str:
        elms = ",".join(map(str, self.lst))
        return f"[{elms}]"


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

y_weight = y_train["Weight"].tolist()
y_waist = y_train["Waist"].tolist()
y_pulse = y_train["Pulse"].tolist()


def fitness_function_lexicase(n: Number):
    cases = []

    cases.append(X_train["Chins"].tolist())
    cases.append(X_train["Situps"].tolist())
    cases.append(X_train["Jumps"].tolist())

    fitnesses = list()

    def fitness_aux(index, y_target):
        y = y_target

        fit = list()

        for c in cases:
            variables = {}
            for x in feature_names:
                i = feature_indices[x]
                variables[x] = c[i]

            # y_pred this wil be a list of 3 functions
            y_pred = n.evaluate(**variables)

            if type(y_pred) in [np.float64, int, float]:
                """If n does not use variables, the output will be scalar."""
                y_pred = np.full(len(y), y_pred)
            try:
                fitness = mean_squared_error(y_pred[index], y[cases.index(c)])
            except InvalidParameterError:
                fitness = BAD_FITNESS
            if isinf(fitness) or np.isnan(fitness):
                fitness = BAD_FITNESS

            fit.append(fitness)

        return sum(fit) / len(fit)

    fitnesses.append(fitness_aux(0, y_weight))
    fitnesses.append(fitness_aux(1, y_waist))
    fitnesses.append(fitness_aux(2, y_pulse))
    return fitnesses


class MultiTargetLexicaseBenchmark:
    def get_grammar(self) -> Grammar:
        return extract_grammar(
            [
                Plus,
                Minus,
                Mul,
                SafeDiv,
                Literal,
                Var,
                SafeSqrt,
                Exp,
                Sin,
                Tanh,
                SafeLog,
            ],
            NumberList,
        )

    def main(self, **args):
        g = self.get_grammar()
        minimizelist = [True for _ in range(X.shape[1])]

        alg = SimpleGP(
            grammar=g,
            minimize=minimizelist,
            fitness_function=fitness_function_lexicase,
            crossover_probability=0.75,
            mutation_probability=0.01,
            max_evaluations=10000,
            max_depth=15,
            population_size=50,
            selection_method=("lexicase",),
            elitism=0,
            **args,
        )
        best = alg.search()[0]
        print(
            f"Fitness of {best.get_fitness(alg.get_problem())} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
        )


if __name__ == "__main__":
    p = MultiTargetLexicaseBenchmark()
    p.main(seed=0)
