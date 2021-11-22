from typing import Generic, Protocol, TypeVar, ForwardRef, Tuple, get_args

from geneticengine.core.random.sources import RandomSource
from geneticengine.metahandlers.base import MetaHandlerGenerator


class VarRange(MetaHandlerGenerator):
    def __init__(self, options):
        self.options = options

    def generate(self, r: RandomSource, receiver, new_symbol, depth, base_type):
        receiver(r.choice(self.options))

    def __repr__(self):
        return str(self.options)
