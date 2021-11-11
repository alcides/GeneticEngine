from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource


class RandomSearch(object):
    def __init__(self, g: Grammar, representation, e):
        self.grammar = g
        self.representation = representation
        self.evaluation = e

    def evolve(self):
        best = 0
        best_ind = None
        for _ in range(1000):
            i = self.representation(RandomSource(), self.grammar, 15)
            f = self.evaluation(i)
            if f > best:
                best = f
                best_ind = i

        return (best_ind, best)
