from typing import List, Tuple


def get_arguments(n) -> List[Tuple[str, type]]:
    if hasattr(n, "__annotations__"):
        args = n.__annotations__
        return [(a, args[a]) for a in args]
    else:
        return []
