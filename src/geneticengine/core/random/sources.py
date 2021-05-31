import random

from typing import List, Any


class RandomSource(object):
    def choice(self, choices: List[Any]):
        i = self.randint(0, len(choices) - 1)
        return choices[i]

    def randint(self, min, max):
        return random.randint(min, max)
