from typing import (
    Any,
    Callable,
    Dict,
    Type,
)

from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator

from geneticengine.core.grammar import Grammar


class VarRange(MetaHandlerGenerator):
    def __init__(self, options):
        if len(options)==0:
            raise Exception(f"Invalid varrange specified: {options}")
        self.options = options

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        context: Dict[str, str],
    ):
        rec(r.choice(self.options))

    def __repr__(self):
        return str(self.options)
