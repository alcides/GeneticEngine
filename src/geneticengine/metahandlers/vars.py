from typing import Generic, Protocol, TypeVar, ForwardRef, Tuple, get_args

from geneticengine.core.random.sources import RandomSource


class VarRange(object):
    def __init__(self, options):
        self.options = options

    def generate(self, r: RandomSource):
        return r.choice(self.options)

    def __repr__(self):
        return str(self.options)