from __future__ import annotations

from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geml.grammars.letter import Char
from geml.grammars.letter import Consonant
from geml.grammars.letter import LetterString
from geml.grammars.letter import String
from geml.grammars.letter import Vowel

# ===================================
# This is a simple example on how to use GeneticEngine to solve a GP problem.
# We define the tree structure of the representation and then we define the fitness function for our problem
# The string match problem aims to find a string that matches the given target string.
# ===================================


# Extracted from PonyGE
def fit(individual: String):
    guess = str(individual)
    target = "Hello world!"
    fitness: float = max(len(target), len(guess))
    # Loops as long as the shorter of two strings
    for t_p, g_p in zip(target, guess):
        if t_p == g_p:
            # Perfect match.
            fitness -= 1
        else:
            # Imperfect match, find ASCII distance to match.
            fitness -= 1 / (1 + (abs(ord(t_p) - ord(g_p))))
    return fitness


def fitness_function(x):
    return fit(x)


class StringMatchBenchmark:
    def get_problem(self) -> Problem:
        return SingleObjectiveProblem(
            minimize=True,
            fitness_function=fitness_function,
        )

    def get_grammar(self) -> Grammar:
        return extract_grammar([LetterString, Char, Vowel, Consonant], String)

    def main(self, **args):
        g = self.get_grammar()
        prob = self.get_problem()
        alg = SimpleGP(
            g,
            problem=prob,
            probability_crossover=0.75,
            probability_mutation=0.01,
            max_depth=10,
            number_of_generations=30,
            population_size=50,
            selection_method=("tournament", 2),
            n_elites=5,
            **args,
        )
        best = alg.evolve()
        print(
            f"Fitness of {best.get_fitness(prob)} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
        )


if __name__ == "__main__":
    StringMatchBenchmark().main(seed=0)
