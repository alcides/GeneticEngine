from abc import ABC
from ast import Num
from dataclasses import dataclass
from typing import Annotated, Any, Callable
import os
import numpy as np
import pandas as pd
from math import isinf

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.grammatical_evolution import ge_representation
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.metahandlers.floats import FloatRange
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metahandlers.ints import IntRange

from geneticengine.metrics import f1_score
from sklearn.metrics import f1_score, accuracy_score


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBLEM = "GenomicsClassification"
DATA_DIR = os.path.join(ROOT_DIR, "data", PROBLEM)

TRAIN_DATA = os.path.join(DATA_DIR, "Train.tsv")
TEST_DATA = os.path.join(DATA_DIR, "Test.tsv")
AUXILIAR_DATASET = os.path.join(DATA_DIR, "MOTIF_MATCHES.tsv.gz")


class DecisionNode(ABC):
    def evaluate(self, X):
        ...

class Number(ABC):
    def evaluate(self, X):
        ...

@dataclass
class And(DecisionNode):
    left: DecisionNode
    right: DecisionNode

    def evaluate(self, X):
        le = self.left.evaluate(X)
        re = self.right.evaluate(X)
        return le & re

    def __str__(self):
        return f"({self.left}) & ({self.right})"

# @dataclass
# class Mul(DecisionNode):
#     left: DecisionNode
#     right: DecisionNode

#     def evaluate(self, X):
#         le = self.left.evaluate(X)
#         re = self.right.evaluate(X)
#         return le * re

#     def __str__(self):
#         return f"({self.left}) & ({self.right})"
    
# @dataclass
# class VarDistancesBetweenMotifs(Number):
#     name: Annotated[str, VarRange([])]

#     def evaluate(self, **kwargs):
#         return kwargs[self.name]

#     def __str__(self) -> str:
#         return self.name

# @dataclass
# class VarDistancesToSpliceSites(Number):
#     name: Annotated[str, VarRange([])]

#     def evaluate(self, **kwargs):
#         return kwargs[self.name]

#     def __str__(self) -> str:
#         return self.name

# # @dataclass
# class VarBoolean(Boolean):
#     name: Annotated[str, VarRange([])]
    
#     def evaluate(self, **kwargs):
#         return kwargs[self.name]

#     def __str__(self) -> str:
#         return self.name

@dataclass
class Threshold(Number):
    value: Annotated[float, FloatRange(0, 1)]
    
    def evaluate(self, X):
        return self.value

    def __str__(self) -> str:
        return f"{self.value}"
    
@dataclass
class Variable(Number):
    name: Annotated[str, VarRange([])]
    
    def evaluate(self, X):
        return X.loc[:, self.name]

        
    def __str__(self) -> str:
        return f"{self.name}"

@dataclass
class LessThan(DecisionNode):
    n1: Variable
    n2: Number

    def evaluate(self, X):
        return self.n1.evaluate(X) < self.n2.evaluate(X)
    
    def __str__(self) -> str:
        return f"{self.n1} < {self.n2}"

# @dataclass
# class TotalCount(Scalar):
#     col_dataset1: Annotated[int, VarRange([])]
#     col_dataset2: Annotated[int, VarRange([])]

#     def count_values(self, i: int, aux_dataset: pd.DataFrame):
#         return np.sum([1 for e in aux_dataset[self.col_dataset2] if e == i])

#     def evaluate(self, X, aux_dataset):
#         return np.array([self.count_values(i, aux_dataset) for i in X[:, self.col_dataset1]])

#     def __str__(self):
#         return f"sum(dataset_secundario[{self.col_dataset2} == dataset_primario[{self.col_dataset1}]])"


# @dataclass
# class And(DecisionNode):
#     left: DecisionNode
#     right: DecisionNode

#     def evaluate(self, X):
#         le = self.left.evaluate(X)
#         re = self.right.evaluate(X)
#         return le & re

#     def __str__(self):
#         return f"({self.left}) + ({self.right})"


# @dataclass
# class LessThanToSpliceSite(DecisionNode):
#     left: IntNode
#     right: Annotated[str, VarRange(['distance_to_donor', 'distance_to_acceptor'])]

#     def evaluate(self, X):
#         arrb = self.left.evaluate(X) < self.right
#         arri = np.array(arrb, dtype=int)
#         return arri

#     def __str__(self):
#         return f"({self.left}) < ({self.right})"


def main():
 
    raw = pd.read_csv(TRAIN_DATA, sep="\t")

    y = raw.iloc[: , -1]
    instance_indexer_col = 0
    X = raw.drop(raw.columns[-1], axis=1)

    feature_names = [c for i, c in enumerate(list(X.columns)) if i != instance_indexer_col]
    feature_indices = {}
    for i, n in enumerate(feature_names):
        feature_indices[n] = i

    AUX = pd.read_csv(AUXILIAR_DATASET, sep="\t")

  
    # Prepare Grammar
    Variable.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
    Variable.feature_indices = feature_indices

    # VarDistancesToSpliceSites.__annotations__["name"] = Annotated[str, VarRange(['distance_to_donor',
    #                                                                              'distance_to_acceptor'])]
    
    # VarDistancesBetweenMotifs.__annotations__["name"] = Annotated[str, VarRange(['Start', 'End',
    #                                                                              'rbp_motif', 'rbp_name_motif'])]
    
    # VarBoolean.__annotations__["name"] = Annotated[str, VarRange(['has_submotif', 
    #                                                               'high_density_region', 
    #                                                               'is_in_exon'])]
    
    g = extract_grammar([And, LessThan, Threshold, Variable], DecisionNode)
    
    def fitness_function(d: DecisionNode):
        print(d)
        y_pred = d.evaluate(X)
        return accuracy_score(y, y_pred)
    
    alg = GP(
        g,
        fitness_function,
        representation=treebased_representation,
        minimize=False,
        seed=122,
        population_size=500,
        number_of_generations=5,
        selection_method=("tournament", 2),
        max_depth=5,
        probability_crossover=0.9,
        n_elites=5,
        target_fitness=1,
        #safe_gen_to_csv=('gen', '')
    )
    (e, fitness, classifier) = alg.evolve(verbose=1)
    print(e)
    print(fitness)
    print(classifier)


main()