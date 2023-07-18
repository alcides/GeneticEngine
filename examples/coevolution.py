from dataclasses import dataclass


def blackbox_classifier(s: str) -> float:
    aapos = s.find("aa")
    ttpos = s.find("tt")
    if aapos == -1 and ttpos == -1:
        return 0
    elif (aapos == -1) or (ttpos == -1):
        return 0.1
    elif aapos > ttpos:
        return 0.2
    else:
        return (ttpos - aapos) / len(s)


@dataclass
class Line:
    str: str  # TODO: length


@dataclass
class Dataset:
    lines: list[Line]  # TODO: length
