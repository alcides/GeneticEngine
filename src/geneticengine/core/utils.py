from typing import List, Tuple


def get_arguments(n) -> List[Tuple[str, type]]:
    args = n.__init__.__annotations__
    return [(a, args[a]) for a in args]