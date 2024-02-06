from abc import ABC

from geneticengine.evaluation.recorder import SingleObjectiveProgressRecorder


class SynthesisAlgorithm(ABC):
    recorder: SingleObjectiveProgressRecorder

    def __init__(self, recorder: SingleObjectiveProgressRecorder):
        self.recorder = recorder
