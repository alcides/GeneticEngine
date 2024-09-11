from __future__ import annotations

from math import isinf
from typing import Annotated, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from geml.simplegp import SimpleGP
from geneticengine.algorithms.hill_climbing import HC
from geneticengine.evaluation.budget import TimeBudget
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.api import Representation
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geml.grammars.basic_math import SafeDiv
from geml.grammars.basic_math import SafeLog
from geml.grammars.basic_math import SafeSqrt
from geml.grammars.literals import exp_literals
from geml.grammars.literals import ExpLiteral
from geml.grammars.sgp import Mul
from geml.grammars.sgp import Number
from geml.grammars.sgp import Plus
from geml.grammars.sgp import Var
from geneticengine.grammar.metahandlers.vars import VarRange
from geml.metrics import mse
from geml.metrics import r2
from geml.sympy_compatible import fix_all


class GeneticProgrammingRegressor(BaseEstimator, TransformerMixin):
    """Genetic Programming Regressor. Main attributes: fit and predict Defaults
    as given in A Field Guide to GP, p.17, by Poli and Mcphee.

    Args:
        nodes (List[Number]): The list of nodes to be used in the grammar. You can design your own, or use the ones in geneticengine.grammars.[sgp,literals,basic_math]. The default uses [ Plus, Mul, ExpLiteral, Var, SafeDiv, SafeLog, SafeSqrt ] + exp_literals.
        representation (Representation): The individual representation used by the GP program. The default is TreeBasedRepresentation. Currently Genetic Engine also supports Grammatical Evolution: geneticengine.representations.grammatical_evolution.GrammaticalEvolutionRepresentation. You can also deisgn your own.
        seed (int): The seed for the RandomSource (default = 123).
        population_size (int): The population size (default = 200). Apart from the first generation, each generation the population is made up of the elites, novelties, and transformed individuals from the previous generation. Note that population_size > (elitism + novelty + 1) must hold.
        elitism (int): Number of elites, i.e. the number of best individuals that are preserved every generation (default = 5).
        novelty (int): Number of novelties, i.e. the number of newly generated individuals added to the population each generation. (default = 10).
        number_of_generations (int): Number of generations (default = 100).
        max_depth (int): The maximum depth a tree can have (default = 15).
        favor_less_complex_trees (bool): If set to True, this gives a tiny penalty to deeper trees to favor simpler trees (default = False).
        hill_climbing (bool): Allows the user to change the standard mutation operations to the hill-climbing mutation operation, in which an individual is mutated to 5 different new individuals, after which the best is chosen to survive (default = False).
        metric (str): Choose the metric used in the fitness function. Currently available: mean square error (mse), and the r2 measure (r2), new once can be requested or easily implemented by the user (default = 'mse').

        mutation_probability (float): probability that an individual is mutated (default = 0.01).
        crossover_probability (float): probability that an individual is chosen for cross-over (default = 0.9).
    """

    def __init__(
        self,
        nodes: list[type[Number]] | None = None,
        representation_name: str = "treebased",
        population_size: int = 200,
        elitism: int = 5,  # Shouldn't this be a percentage of population size?
        novelty: int = 10,
        max_depth: int = 15,
        seed: int = 123,
        metric: str = "mse",
        # Budget:
        time_budget: int = 30,
        number_of_generations: int = 100,
        # -----
        # As given in A Field Guide to GP, p.17, by Poli and Mcphee
        mutation_probability: float = 0.01,
        crossover_probability: float = 0.9,
        selection_method=("tournament", 5),
        initial_population: list[Any] | None = None,
        # -----
        parallel: bool = False,
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
        assert population_size > (elitism + novelty + 1)
        assert Var in nodes

        assert population_size > (elitism + novelty + 1)
        assert Var in nodes

        self.representation_name = representation_name
        self.seed = seed
        self.random = NativeRandomSource(self.seed)

        self.nodes = nodes
        self.max_depth = max_depth

        self.metric_name = metric

        self.initial_population = initial_population
        self.population_size = population_size
        self.elitism = elitism
        self.novelty = novelty
        self.selection_method = selection_method

        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.parallel_evaluation = parallel

        self.evaluations_budget = number_of_generations * population_size
        self.time_budget = time_budget

    def fit(self, X, y):
        """Fits the regressor with data X and target y."""
        if isinstance(y, pd.Series):
            target = y.values
        else:
            target = y

        if isinstance(X, pd.DataFrame):
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

        if self.metric_name == "r2":
            metric = r2
            minimise = False
        else:
            metric = mse
            minimise = True

        def fitness_function(n: Number):
            variables = {}
            for x in feature_names:
                i = feature_indices[x]
                variables[x] = data[:, i]
            y_pred = n.evaluate(**variables)

            fitness = metric(
                y_pred,
                target,
            )
            if isinf(fitness) or np.isnan(fitness):
                fitness = 100000000
            return fitness

        model = SimpleGP(
            fitness_function=fitness_function,
            grammar=self.grammar,
            minimize=minimise,
            representation=self.representation_name,
            max_depth=self.max_depth,
            target_fitness=0,
            max_time=self.time_budget,
            max_evaluations=self.evaluations_budget,
            parallel_evaluation=self.parallel_evaluation,
            seed=self.seed,
            population_size=self.population_size,
            elitism=self.elitism,
            novelty=self.novelty,
            initial_population=self.initial_population,
            mutation_probability=self.mutation_probability,
            crossover_probability=self.crossover_probability,
            selection_method=self.selection_method,
        )

        ind = model.search()
        self.evolved_phenotype = ind.get_phenotype()
        self.sympy_compatible_phenotype = fix_all(str(self.evolved_phenotype))

    def predict(self, X):
        """Predict the target values of X.

        The model must have been fitted
        """
        assert self.evolved_phenotype is not None
        if (isinstance(X, pd.DataFrame)) or (isinstance(X, pd.Series)):
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

    def score(self, X, y):
        y_pred = self.predict(X)

        return r2(y_pred, y)


class HillClimbingRegressor(BaseEstimator, TransformerMixin):
    """Hill Climbing Regressor. Main attributes: fit and predict.

    Args:
        nodes (List[Number]): The list of nodes to be used in the grammar. You can design your own, or use the ones in geneticengine.grammars.[sgp,literals,basic_math]. The default uses [ Plus, Mul, ExpLiteral, Var, SafeDiv, SafeLog, SafeSqrt ] + exp_literals.
        representation (Representation): The individual representation used by the GP program. The default is TreeBasedRepresentation. Currently Genetic Engine also supports Grammatical Evolution: geneticengine.representations.grammatical_evolution.GrammaticalEvolutionRepresentation. You can also deisgn your own.
        seed (int): The seed for the RandomSource (default = 123).
        population_size (int): The population size (default = 200). Apart from the first generation, each generation the population is made up of the elites, novelties, and transformed individuals from the previous generation. Note that population_size > (elitism + novelty + 1) must hold.
        number_of_generations (int): Number of generations (default = 100).
        max_depth (int): The maximum depth a tree can have (default = 15).
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
        representation_class: type[Representation] = TreeBasedRepresentation,
        population_size: int = 200,
        number_of_generations: int = 100,
        max_depth: int = 15,
        seed: int = 123,
        metric: str = "mse",
    ):
        assert Var in nodes

        self.nodes = nodes
        self.representation_class = representation_class
        self.evolved_ind = None
        self.nodes = nodes
        self.random = NativeRandomSource(seed)
        self.seed = seed
        self.population_size = population_size
        self.max_depth = max_depth
        self.number_of_generations = number_of_generations
        if metric == "r2":
            self.metric = r2
            self.minimise = False
        else:
            self.metric = mse
            self.minimise = True

    def fit(self, X, y):
        """Fits the regressor with data X and target y."""
        if isinstance(y, pd.Series):
            target = y.values
        else:
            target = y

        if isinstance(X, pd.DataFrame):
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

            fitness = self.metric(
                y_pred,
                target,
            )
            if isinf(fitness) or np.isnan(fitness):
                fitness = 100000000
            return fitness

        r = NativeRandomSource(self.seed)
        model = HC(
            problem=SingleObjectiveProblem(
                minimize=False,
                fitness_function=fitness_function,
            ),
            budget=TimeBudget(3),
            representation=self.representation_class(self.grammar, MaxDepthDecider(r, self.grammar, self.max_depth)),
            random=r,
        )

        ind = model.search()
        self.evolved_phenotype = ind.get_phenotype()
        self.sympy_compatible_phenotype = fix_all(str(self.evolved_phenotype))

    def predict(self, X):
        """Predict the target values of X.

        The model must have been fitted
        """
        assert self.evolved_phenotype is not None
        if (isinstance(X, pd.DataFrame)) or (isinstance(X, pd.Series)):
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
