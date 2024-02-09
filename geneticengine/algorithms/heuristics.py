from __future__ import annotations

from geneticengine.algorithms.api import SynthesisAlgorithm

from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.recorder import SingleObjectiveProgressTracker
from geneticengine.problems import Problem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation


class HeuristicSearch(SynthesisAlgorithm):
    """Randomly generates new solutions and keeps the best one."""

    random: RandomSource

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: Representation,
        random: RandomSource = None,
        recorder: SingleObjectiveProgressTracker | None = None,
    ):
        super().__init__(problem, budget, representation, recorder)
        if random is None:
            self.random = NativeRandomSource(0)
        else:
            self.random = random
