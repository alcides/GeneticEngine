from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import Annotated
from typing import List
from typing import Type

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.hill_climbing import HC
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.basic_math import SafeDiv
from geneticengine.grammars.basic_math import SafeLog
from geneticengine.grammars.basic_math import SafeSqrt
from geneticengine.grammars.literals import exp_literals
from geneticengine.grammars.literals import ExpLiteral
from geneticengine.grammars.sgp import Literal
from geneticengine.grammars.sgp import Mul
from geneticengine.grammars.sgp import Number
from geneticengine.grammars.sgp import Plus
from geneticengine.grammars.sgp import Var
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics import f1_score
from geneticengine.metrics import mse
from geneticengine.off_the_shelf.sympy_compatible import fix_all


class GeneticProgrammingRegressor(BaseEstimator, TransformerMixin):
    """
    Genetic Programming Regressor. Main attributes: fit and predict

    Parameters:
        - nodes (List[Number]): The list of nodes to be used in the grammar. You can design your own, or use the ones in geneticengine.grammars.[sgp,literals,basic_math]. The default uses [ Plus, Mul, ExpLiteral, Var, SafeDiv, SafeLog, SafeSqrt ] + exp_literals.
        - representation (Representation): The individual representation used by the GP program. The default is treebased_representation. Currently Genetic Engine also supports Grammatical Evolution: geneticengine.core.representations.grammatical_evolution.ge_representation. You can also deisgn your own.
        - seed (int): The seed for the RandomSource (default = 123).
        - population_size (int): The population size (default = 200). Apart from the first generation, each generation the population is made up of the elites, novelties, and transformed individuals from the previous generation. Note that population_size > (n_elites + n_novelties + 1) must hold.
        - n_elites (int): Number of elites, i.e. the number of best individuals that are preserved every generation (default = 5).
        - n_novelties (int): Number of novelties, i.e. the number of newly generated individuals added to the population each generation. (default = 10).
        - number_of_generations (int): Number of generations (default = 100).
        - max_depth (int): The maximum depth a tree can have (default = 15).
        - favor_less_deep_trees (bool): If set to True, this gives a tiny penalty to deeper trees to favor simpler trees (default = False).
        - hill_climbing (bool): Allows the user to change the standard mutation operations to the hill-climbing mutation operation, in which an individual is mutated to 5 different new individuals, after which the best is chosen to survive (default = False).
        -----
        Defaults as given in A Field Guide to GP, p.17, by Poli and Mcphee:
        - probability_mutation (float): probability that an individual is mutated (default = 0.01).
        - probability_crossover (float): probability that an individual is chosen for cross-over (default = 0.9).
        -----
    """

    def __init__(
        self,
        nodes: list[type[Number]] = None,
        representation: Representation = treebased_representation,
        population_size: int = 200,
        n_elites: int = 5,  # Shouldn't this be a percentage of population size?
        n_novelties: int = 10,
        number_of_generations: int = 100,
        max_depth: int = 15,
        favor_less_deep_trees: bool = True,
        hill_climbing: bool = False,
        seed: int = 123,
        # -----
        # As given in A Field Guide to GP, p.17, by Poli and Mcphee
        probability_mutation: float = 0.01,
        probability_crossover: float = 0.9,
        # -----
    ):
        if nodes is None:
            nodes = [
                Plus,
                Mul,
                ExpLiteral,
                Var,
                SafeDiv,
                SafeLog,
                SafeSqrt,
                *exp_literals,
            ]
        assert population_size > (n_elites + n_novelties + 1)
        assert Var in nodes

        self.nodes = nodes
        self.representation = representation
        self.evolved_ind = None
        self.nodes = nodes
        self.random = RandomSource(seed)
        self.seed = seed
        self.population_size = population_size
        self.max_depth = max_depth
        self.favor_less_deep_trees = favor_less_deep_trees
        self.n_elites = n_elites
        self.n_novelties = n_novelties
        self.hill_climbing = hill_climbing
        self.number_of_generations = number_of_generations
        self.probability_mutation = probability_mutation
        self.probability_crossover = probability_crossover

    def fit(self, X, y):
        """
        Fits the regressor with data X and target y.
        """
        if type(y) == pd.Series:
            target = y.values
        else:
            target = y

        if type(X) == pd.DataFrame:
            feature_names = list(X.columns.values)
            data = X.values
        else:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
            data = X
        feature_indices = {}
        for i, n in enumerate(feature_names):
            feature_indices[n] = i

        Var.__init__.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
        Var.feature_indices = feature_indices

        self.grammar = extract_grammar(self.nodes, Number)
        self.feature_names = feature_names
        self.feature_indices = feature_indices

        def fitness_function(n: Number):
            variables = {}
            for x in feature_names:
                i = feature_indices[x]
                variables[x] = data[:, i]
            y_pred = n.evaluate(**variables)

            fitness = mse(
                y_pred,
                y,
            )  # mse is used in PonyGE, as the error metric is not None!
            if isinf(fitness) or np.isnan(fitness):
                fitness = 100000000
            return fitness

        model = GP(
            grammar=self.grammar,
            evaluation_function=fitness_function,
            minimize=True,
            representation=self.representation,
            population_size=self.population_size,
            n_elites=self.n_elites,
            n_novelties=self.n_novelties,
            number_of_generations=self.number_of_generations,
            max_depth=self.max_depth,
            favor_less_deep_trees=self.favor_less_deep_trees,
            probability_mutation=self.probability_mutation,
            probability_crossover=self.probability_crossover,
            hill_climbing=self.hill_climbing,
            seed=self.seed,
        )

        best_ind, fitness, phenotype = model.evolve(verbose=1)
        self.evolved_phenotype = phenotype
        self.sympy_compatible_phenotype = fix_all(str(phenotype))

    def predict(self, X):
        """
        Predict the target values of X. The model must have been fitted
        """
        assert self.evolved_phenotype != None
        if (type(X) == pd.DataFrame) or (type(X) == pd.Series):
            data = X.values
        else:
            data = X
        if len(data.shape) == 1:
            data = np.array([data])
        assert data.shape[1] == len(self.feature_names)

        variables = {}
        for x in self.feature_names:
            i = self.feature_indices[x]
            variables[x] = data[:, i]
        y_pred = self.evolved_phenotype.evaluate(**variables)

        return y_pred


class HillClimbingRegressor(BaseEstimator, TransformerMixin):
    """
    Hill Climbing Regressor. Main attributes: fit and predict

    Parameters:
        - nodes (List[Number]): The list of nodes to be used in the grammar. You can design your own, or use the ones in geneticengine.grammars.[sgp,literals,basic_math]. The default uses [ Plus, Mul, ExpLiteral, Var, SafeDiv, SafeLog, SafeSqrt ] + exp_literals.
        - representation (Representation): The individual representation used by the GP program. The default is treebased_representation. Currently Genetic Engine also supports Grammatical Evolution: geneticengine.core.representations.grammatical_evolution.ge_representation. You can also deisgn your own.
        - seed (int): The seed for the RandomSource (default = 123).
        - population_size (int): The population size (default = 200). Apart from the first generation, each generation the population is made up of the elites, novelties, and transformed individuals from the previous generation. Note that population_size > (n_elites + n_novelties + 1) must hold.
        - number_of_generations (int): Number of generations (default = 100).
        - max_depth (int): The maximum depth a tree can have (default = 15).
    """

    def __init__(
        self,
        nodes=[
            Plus,
            Mul,
            ExpLiteral,
            Var,
            SafeDiv,
            SafeLog,
            SafeSqrt,
            *exp_literals,
        ],  # "type: ignore"
        representation: Representation = treebased_representation,
        population_size: int = 200,
        number_of_generations: int = 100,
        max_depth: int = 15,
        seed: int = 123,
    ):
        assert Var in nodes

        self.nodes = nodes
        self.representation = representation
        self.evolved_ind = None
        self.nodes = nodes
        self.random = RandomSource(seed)
        self.seed = seed
        self.population_size = population_size
        self.max_depth = max_depth
        self.number_of_generations = number_of_generations

    def fit(self, X, y):
        """
        Fits the regressor with data X and target y.
        """
        if type(y) == pd.Series:
            target = y.values
        else:
            target = y

        if type(X) == pd.DataFrame:
            feature_names = list(X.columns.values)
            data = X.values
        else:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
            data = X
        feature_indices = {}
        for i, n in enumerate(feature_names):
            feature_indices[n] = i

        Var.__init__.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
        Var.feature_indices = feature_indices

        self.grammar = extract_grammar(self.nodes, Number)
        self.feature_names = feature_names
        self.feature_indices = feature_indices

        def fitness_function(n: Number):
            variables = {}
            for x in feature_names:
                i = feature_indices[x]
                variables[x] = data[:, i]
            y_pred = n.evaluate(**variables)

            fitness = mse(y_pred, y)
            if isinf(fitness) or np.isnan(fitness):
                fitness = 100000000
            return fitness

        model = HC(
            g=self.grammar,
            evaluation_function=fitness_function,
            minimize=True,
            representation=self.representation,
            population_size=self.population_size,
            number_of_generations=self.number_of_generations,
            max_depth=self.max_depth,
            seed=self.seed,
        )

        best_ind, fitness, phenotype = model.evolve(verbose=1)
        self.evolved_phenotype = phenotype
        self.sympy_compatible_phenotype = fix_all(str(phenotype))

    def predict(self, X):
        """
        Predict the target values of X. The model must have been fitted
        """
        assert self.evolved_phenotype != None
        if (type(X) == pd.DataFrame) or (type(X) == pd.Series):
            data = X.values
        else:
            data = X
        if len(data.shape) == 1:
            data = np.array([data])
        assert data.shape[1] == len(self.feature_names)

        variables = {}
        for x in self.feature_names:
            i = self.feature_indices[x]
            variables[x] = data[:, i]
        y_pred = self.evolved_phenotype.evaluate(**variables)

        return y_pred

