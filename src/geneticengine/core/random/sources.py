import random

from typing import List, Any


class RandomSource(object):
    def __init__(self, seed: int = 0):
        self.random = random.Random(seed)

    def choice(self, choices: List[Any]):
        i = self.random.randint(0, len(choices) - 1)
        return choices[i]

    def randint(self, min, max):
        return self.random.randint(min, max)
