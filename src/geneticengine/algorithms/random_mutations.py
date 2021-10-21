from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.treebased import mutate, random_individual


class RandomMutations(object):
    def __init__(self, g: Grammar, representation, e):
        self.grammar = g
        self.representation = representation
        self.evaluation = e
        self.random = RandomSource()

    def evolve(self):
        best = 0
        best_ind = None
        i = random_individual(self.random, self.grammar, 10) # Puts in grammar but random_individual takes Processed Grammar
        for _ in range(1000):
            i = mutate(self.random, self.grammar, i)
            f = self.evaluation(i)
            if f > best:
                print("BESTED")
                best = f
                best_ind = i

        return (best_ind, best)
