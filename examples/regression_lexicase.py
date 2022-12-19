from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.ge import (
    ge_representation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    sge_representation,
)
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.basic_math import Exp
from geneticengine.grammars.basic_math import SafeDiv
from geneticengine.grammars.basic_math import SafeLog
from geneticengine.grammars.basic_math import SafeSqrt
from geneticengine.grammars.basic_math import Sin
from geneticengine.grammars.basic_math import Tanh
from geneticengine.grammars.sgp import Minus
from geneticengine.grammars.sgp import Mul
from geneticengine.grammars.sgp import Number
from geneticengine.grammars.sgp import Plus
from geneticengine.grammars.sgp import Var
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics import mse

DATASET_NAME = "Vladislavleva4"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.txt"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.txt"

bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter="\t")
target = bunch.response
data = bunch.drop(["response"], axis=1)

feature_names = list(data.columns.values)
feature_indices = {}
for i, n in enumerate(feature_names):
    feature_indices[n] = i

# Prepare Grammar
Var.__init__.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
Var.feature_indices = feature_indices  # type: ignore


@dataclass
class Literal(Number):
    val: Annotated[int, IntRange(0, 9)]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


def preprocess():
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
        Number,
    )

    # <e>  ::=  <e>+<e>|
    #       <e>-<e>|
    #       <e>*<e>|
    #       pdiv(<e>,<e>)|
    #       psqrt(<e>)|
    #       np.sin(<e>)|
    #       np.tanh(<e>)|
    #       np.exp(<e>)|
    #       plog(<e>)|
    #       x[:, 0]|x[:, 1]|x[:, 2]|x[:, 3]|x[:, 4]|
    #       <c>
    # <c>  ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9

def lexicase_parameters():
    X = data.values
    n_cases = 50
    case_size = int(len(X) / n_cases)
    if len(X) % case_size == 0:
        minimize_list = [True for _ in range(n_cases)]
    else:
        minimize_list = [True for _ in range(n_cases + 1)]
            
    def lexicase_fitness_function(n: Number):
        X = data.values
        y = target.values

        variables = {}
        for x in feature_names:
            i = feature_indices[x]
            variables[x] = X[:, i]

        try:
            y_pred = n.evaluate(**variables)
            pred_error = np.power(y_pred - y, 2)
            grouped_errors = list()
            for i in range(n_cases):
                grouped_errors.append(sum(pred_error[case_size*i:case_size*(i+1)])/len(pred_error[case_size*i:case_size*(i+1)]))
            if len(X) % case_size != 0:
                grouped_errors.append(sum(pred_error[(case_size*n_cases):])/len(pred_error[(case_size*n_cases):]))
        except (OverflowError, ValueError) as e:
            return np.full(len(y), 99999999999)
        return grouped_errors
        
    return lexicase_fitness_function, minimize_list


def evolve(
    g,
    seed,
    mode,
    representation="treebased_representation",
):
    if representation == "ge":
        representation = ge_representation
    elif representation == "sge":
        representation = sge_representation
    else:
        representation = treebased_representation

    fitness_function_lexicase, minimizelist = lexicase_parameters()

    def single_criteria_test(ind) -> float:
        return sum((m and -f or f) for (f, m) in zip(ind.fitness, minimizelist))

    alg = GP(
        g,
        representation=representation,
        problem=MultiObjectiveProblem(
            minimize=minimizelist,
            fitness_function=fitness_function_lexicase,
            best_individual_criteria_function=single_criteria_test,
        ),
        probability_crossover=0.75,
        probability_mutation=0.01,
        number_of_generations=10,
        max_depth=8,
        population_size=50,
        selection_method=("lexicase","epsilon"),
        n_elites=0,
        seed=seed,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    b, bf = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {sum(bf)/len(bf)}")
