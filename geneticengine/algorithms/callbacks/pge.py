from __future__ import annotations

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.individual import Individual


class PGECallback(Callback):
    """This callback updates the weight of a probabilistic grammar to bias
    towards the best individuals."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def process_iteration(
        self,
        generation: int,
        population: list[Individual],
        time: float,
        gp: GP,
    ):
        best = gp.get_best_individual(gp.problem, population)
        prob = best.production_probabilities(
            gp.representation.genotype_to_phenotype, gp.representation.grammar
        )
        gp.representation.grammar = gp.representation.grammar.update_weights(
            self.learning_rate,
            prob,
        )
