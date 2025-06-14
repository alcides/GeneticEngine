from __future__ import annotations


from examples.benchmarks.benchmark import Benchmark, example_run
from geml.grammars.letter import Char, Consonant, LetterString, String, Vowel
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem


class StringMatchBenchmark(Benchmark):
    def __init__(self, target: str = "Hellow world!"):
        self.setup_problem(target)
        self.setup_grammar()

    def setup_problem(self, target):

        # Problem
        def fitness_function(individual) -> float:
            guess = str(individual)
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

        self.problem = SingleObjectiveProblem(minimize=True, fitness_function=fitness_function, target=0)

    def setup_grammar(self):
        self.grammar = extract_grammar([LetterString, Char, Vowel, Consonant], String)

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    example_run(StringMatchBenchmark("Hello world!"))
