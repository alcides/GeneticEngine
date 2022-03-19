from __future__ import annotations

from abc import ABC


class Callback(ABC):
    def process_iteration(self, generation: int, population, time: float):
        ...


class DebugCallback(Callback):
    """Example of a callback that prints all the individuals in the population"""

    def process_iteration(self, generation: int, population, time: float):
        for p in population:
            print(p)
        print(f"___ end of gen {generation}")
