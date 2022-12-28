from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.individual import Individual


class ElitismDebugCallback(Callback):
    """Crashes the program if the fitness decreases."""

    def __init__(self):
        self.bests = []

    def process_iteration(self, generation: int, population, time: float, gp):
        gp.evaluator.eval(gp.problem, population)
        best = max(population, key=Individual.key_function(gp.problem))
        best = Individual.key_function(gp.problem)(best)
        if self.bests:
            assert best >= self.bests[-1]
        self.bests.append(best)

    def end_evolution(self):
        print("Final best fitnesses", self.bests)
