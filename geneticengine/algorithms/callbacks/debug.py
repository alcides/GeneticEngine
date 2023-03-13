from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.core.fitness_helpers import best_individual, is_better


class ElitismDebugCallback(Callback):
    """Crashes the program if the fitness decreases."""

    def __init__(self):
        self.bests = []

    def process_iteration(self, generation: int, population, time: float, gp):
        gp.evaluator.eval(gp.problem, population)
        best = best_individual(population=population, problem=gp.problem)
        assert is_better(gp.problem, best, self.bests[-1])
        self.bests.append(best)

    def end_evolution(self):
        print("Final best fitnesses", self.bests)
