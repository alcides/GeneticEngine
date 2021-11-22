from typing import Protocol

from geneticengine.core.random.sources import RandomSource


class MetaHandlerGenerator(Protocol):
    def generate(self, r: RandomSource, receiver, new_symbol, depth, base_type):
        ...
