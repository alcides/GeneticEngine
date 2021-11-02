from geneticengine.core.utils import get_arguments


class Node(object):
    depth = None
    distance_to_term = None
    nodes = None
    
    def evaluate(self, **kwargs):
        return None

    def __repr__(self):
        args = ", ".join(
            [f"{a}={getattr(self, a)}" for (a, at) in get_arguments(self.__class__)]
        )
        return f"{self.__class__.__name__}({args})"