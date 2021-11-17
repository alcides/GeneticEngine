import string
from abc import ABC
from dataclasses import dataclass
from textwrap import indent
from typing import Annotated

from geneticengine.metahandlers.vars import VarRange
from geneticengine.core.grammar import extract_grammar
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.representations.treebased import treebased_representation

# Auxiliary "attributes"
an_char = list(string.digits) + list(string.ascii_letters)
s_char = [
    "!", "#", "$", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";",
    "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "}", "~", "|",
    '\"', "'", " "
]


# re ::= elementary-re re | elementary-re
@dataclass
class RE(ABC):
    pass


# elementary-re ::= set | range | modifier | char | {match_times} | lookarounds
@dataclass
class ElementaryRE(RE):
    pass


# elementary-re ::= [RE] | (RE)
class ElementaryREParens(RE):
    option: Annotated[str, VarRange(['[{}]', '({})'])]
    regex: RE

    def __str__(self):
        return self.option.format(str(self.regex))


# elementary-re ::= \w | \d
class ElementaryREWD(RE):
    option: Annotated[str, VarRange(['\w', '\d'])]

    def __str__(self):
        return self.option


# Elementary-RE RE
@dataclass
class ElementaryRERE(RE):
    elementary_regex: ElementaryRE
    regex: RE

    def __str__(self):
        return f'{str(self.elementary_regex)}{self.regex}'


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


# set ::= char | char set
@dataclass
class Set(ElementaryRE):
    pass


# char ::= s_char | an_char | an_char | an_char
@dataclass
class Char(Set):
    character: Annotated[str, VarRange(s_char + an_char * 3)]

    def __str__(self):
        return self.character


# char ::= char set
@dataclass
class SetChar(Set):
    character: Char
    _set: Set

    def __str__(self):
        return f'{self.char}{self._set}'


@dataclass
class Range(ElementaryRE):
    pass
