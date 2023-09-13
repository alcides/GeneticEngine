from __future__ import annotations

from geneticengine.algorithms.callbacks.callback import Callback


class GrammarDebugCallback(Callback):
    def __init__(self, g):
        self.grammar = g

    def process_iteration(self, generation: int, population, time: float, gp):
        print("----")
        for ind in population:
            ind.evaluate(gp.problem)
            print(ind.genotype, ind.phenotype, ind.fitness)
        print(self.grammar)
        print(".....")

    def end_evolution(self):
        pass
