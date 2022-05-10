from typing import List, Callable, Any


class Evaluator(object):
    def eval(self, f: Callable[[Any], float], indivs: List[Any]):
        raise NotImplementedError()


class SeqEvaluator(Evaluator):
    def eval(self, f: Callable[[Any], float], indivs: List[Any]):
        return list(map(f, indivs))
