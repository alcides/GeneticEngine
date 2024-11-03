from __future__ import annotations

import abc
import copy
from dataclasses import dataclass
from typing import Annotated, Any

from geml.simplegp import SimpleGP
from geneticengine.grammar.decorators import weight
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.vars import VarRange


# ===================================
# This is a simple example on how to use GeneticEngine to solve a GP problem.
# We define the tree structure of the representation and then we define the fitness function for our problem
# In this example we are solving a recurrence problem using normal GP
# ===================================


# This dataset will define the problem complexity:

N = 40
dataset = [1, 1]
for i in range(N):
    dataset.append(dataset[-1] + dataset[-2])


# Now we define the structure of the tree. Metahandlers (Annotated types) will be able to restrict the domain of elements.


class Node(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, input: list[int]): ...


@dataclass
class Op(Node):
    right: Node
    op: Annotated[str, VarRange(["+", "-", "*", "/"])]
    left: Node

    def evaluate(self, input: list[int]):
        if self.op == "+":
            return self.right.evaluate(input) + self.left.evaluate(input)
        elif self.op == "-":
            return self.right.evaluate(input) - self.left.evaluate(input)
        elif self.op == "*":
            return self.right.evaluate(input) * self.left.evaluate(input)
        else:
            if self.left.evaluate(input) == 0:
                return 0
            return self.right.evaluate(input) // self.left.evaluate(input)

    def __str__(self):
        return f"({self.right} {self.op} {self.left})"


@dataclass
class Access(Node):
    i: Node  # This could be a literal, for better performance
    # but no higher order operator

    def evaluate(self, input: list[int]):
        v = self.i.evaluate(input)

        return input[v % len(input)]

    def __str__(self):
        return f"x@{self.i}"


@dataclass
class Literal(Node):
    i: Annotated[int, IntRange(-10, 10)]

    def evaluate(self, input: list[int]):
        return self.i

    def __str__(self):
        return f"{self.i}"


@weight(99999)
@dataclass
class KnowledgeLiteral(Node):
    i: Annotated[int, IntRange(-2, -1)]

    def evaluate(self, input: list[int]):
        return self.i

    def __str__(self):
        return f"{self.i}"


# The fitness function is the other


def fitness_function(p):
    p = copy.deepcopy(p)

    errors = 0
    input = [1, 1]
    for i in range(2, len(dataset)):
        prediction = p.evaluate(input)
        if abs(prediction) > 100000000:
            prediction = 0

        expected = dataset[i]
        e = abs(prediction - expected)
        errors += e**2
        input.append(prediction)
    return errors


if __name__ == "__main__":
    g = extract_grammar([Op, Access, Literal, KnowledgeLiteral], Node)
    gp = SimpleGP(
        grammar=g,
        fitness_function=fitness_function,
        minimize=True,
        max_depth=5,
        max_evaluations=100 * 100,
        population_size=100,
        mutation_probability=0.5,
        crossover_probability=0.4,
        target_fitness=0,
        novelty=10,
    )
    best: Any = gp.search()[0]
    print("Number of nodes:", best.get_phenotype().gengy_nodes)
    print("Distance to Terminal:", best.get_phenotype().gengy_distance_to_term)
    fitness = best.get_fitness(gp.get_problem())
    print(
        f"Fitness of {fitness} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
    )

    print("Dataset size:", len(dataset))
    print(fitness_function(best.get_phenotype()))
