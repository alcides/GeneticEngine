from abc import ABC
from dataclasses import dataclass
from textwrap import indent
from typing import Annotated

from geneticengine.metahandlers.vars import VarRange
from geneticengine.core.grammar import extract_grammar
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.representations.treebased import treebased_representation


# re ::= elementary-re re | elementary-re
@dataclass
class RE(ABC):
    pass


# elementary-re ::= [RE] | (RE) | set | range | modifier | char | \w | \d | {match_times} | lookarounds
@dataclass
class ElementaryRE(RE):
    pass


# Elementary-RE RE
@dataclass
class ElementaryRERE(RE):
    elementary_regex: ElementaryRE
    regex: RE

    pass


# modifier ::= ^RE | RE. | RE* | RE+ | RE++ | RE? | RE?+ |  RE "|" RE
@dataclass
class Modifier(ElementaryRE):
    pass


# modifierSingle ::= ^RE | RE. | RE* | RE+ | RE++ | RE? | RE?+
@dataclass
class ModifierSingle(Modifier):
    modifier: Annotated[str, VarRange(['^', '.', '*', '+', '++', '?', '?+'])]
    regex: RE

    def __str__(self):
        result = str(self.regex)

        if self.modifier == '^':
            result = f'^{result}'
        else:
            result = f'{result}{self.modifier}'

        return result


# modifierOr ::= RE "|" RE
@dataclass
class ModifierOr(Modifier):
    regex1: RE
    regex2: RE

    def __str__(self):
        return f'{self.regex1}|{self.regex2}'


# lookarounds ::= (?<=RE) | (?<!RE) | (?=RE) | (?!RE) | (?:RE) | RE {RE,RE}+
@dataclass
class Lookaround(ElementaryRE):
    pass


# lookaroundSingle ::= (?<=RE) | (?<!RE) | (?=RE) | (?!RE) | (?:RE)
@dataclass
class LookaroundSingle(Lookaround):
    lookaround: Annotated[str, VarRange(['?<=', '?<!', '?=', '?!', '?:'])]
    regex: RE

    def __str__(self):
        return f'({self.lookaround}{self.regex})'


# lookaroundComposition ::= RE {RE,RE}+
@dataclass
class LookaroundComposition(Lookaround):
    regex1: RE
    regex2: RE
    regex3: RE

    def __str__(self):
        return f'{self.regex1}{{{self.regex2},{self.regex3}}}+'
