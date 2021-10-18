from typing import List, Tuple


def get_arguments(n) -> List[Tuple[str, type]]:
    if hasattr(n, "__annotations__"):
        args = n.__annotations__
        return [(a, args[a]) for a in args]
    else:
        return []


def isNonTerminal(bty):
    if hasattr(bty, "__bases__"):
        for ty in bty.__bases__:
            if hasattr(ty, "_is_protocol"):
                return True
    return False


def isTerminal(n) -> bool:
    for (a, at) in get_arguments(n):
        if isNonTerminal(at):
            return False
    return True