from abc import ABC


class Callback(ABC):
    def process_iteration(self, generation: int, population, time: float):
        ...
