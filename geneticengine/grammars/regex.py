from __future__ import annotations

import string
from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.core.decorators import abstract
from geneticengine.metahandlers.vars import VarRange

# Auxiliary "attributes"
an_char = list(string.digits) + list(string.ascii_letters)
s_char = [
    "!",
    "#",
    "$",
    "%",
    "&",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "{",
    "}",
    "~",
    "|",
    '"',
    "'",
    " ",
]


# re ::= elementary-re re | elementary-re
@dataclass
class RE(ABC):
    pass


# elementary-re ::= set | range | modifier | char | {match_times} | lookarounds
@abstract
@dataclass
class ElementaryRE(RE):
    pass


# elementary-re ::= [RE] | (RE)
@dataclass
class ElementaryREParens(RE):
    option: Annotated[str, VarRange(["[{}]", "({})"])]
    regex: RE

    def __str__(self):
        return self.option.format(str(self.regex))


# elementary-re ::= \w | \d
@dataclass
class ElementaryREWD(RE):
    option: Annotated[str, VarRange([r"\w", r"\d"])]

    def __str__(self):
        return self.option


# Elementary-RE RE
@dataclass
class ElementaryRERE(RE):
    elementary_regex: ElementaryRE
    regex: RE

    def __str__(self):
        return f"{str(self.elementary_regex)}{self.regex}"


# modifier ::= ^RE | RE. | RE* | RE+ | RE++ | RE? | RE?+ |  RE "|" RE
@dataclass
@abstract
class Modifier(ElementaryRE):
    pass


# modifierSingle ::= ^RE | RE. | RE* | RE+ | RE++ | RE? | RE?+
@dataclass
class ModifierSingle(Modifier):
    modifier: Annotated[str, VarRange(["^", ".", "*", "+", "++", "?", "?+"])]
    regex: RE

    def __str__(self):
        result = str(self.regex)

        if self.modifier == "^":
            result = f"^{result}"
        else:
            result = f"{result}{self.modifier}"

        return result


# modifierOr ::= RE "|" RE
@dataclass
class ModifierOr(Modifier):
    regex1: RE
    regex2: RE

    def __str__(self):
        return f"{self.regex1}|{self.regex2}"


# lookarounds ::= (?<=RE) | (?<!RE) | (?=RE) | (?!RE) | (?:RE) | RE {RE,RE}+
@abstract
@dataclass
class Lookaround(ElementaryRE):
    pass


# lookaroundSingle ::= (?<=RE) | (?<!RE) | (?=RE) | (?!RE) | (?:RE)
@dataclass
class LookaroundSingle(Lookaround):
    lookaround: Annotated[str, VarRange(["?<=", "?<!", "?=", "?!", "?:"])]
    regex: RE

    def __str__(self):
        return f"({self.lookaround}{self.regex})"


# lookaroundComposition ::= RE {RE,RE}+
@dataclass
class LookaroundComposition(Lookaround):
    regex1: RE
    regex2: RE
    regex3: RE

    def __str__(self):
        return f"{self.regex1}{{{self.regex2},{self.regex3}}}+"


# set ::= char | char set
@abstract
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
        return f"{self.char}{self._set}"


# range ::=  an_char - an_char | an_char - an_char | A-Z | a-z | 0-9
@abstract
@dataclass
class Range(ElementaryRE):
    pass


@dataclass
class RangeAnChar1(Range):
    character1: Annotated[str, VarRange(an_char)]
    character2: Annotated[str, VarRange(an_char)]

    def __str__(self):
        return f"{self.character1}-{self.character2}"


@dataclass
class RangeAnChar2(Range):
    character1: Annotated[str, VarRange(an_char)]
    character2: Annotated[str, VarRange(an_char)]

    def __str__(self):
        return f"{self.character1}-{self.character2}"


@dataclass
class RangeLimits(Range):
    option: Annotated[str, VarRange(["A-Z", "a-z", "0-9"])]

    def __str__(self):
        return self.option


# match_times ::= recur_digit | recur_digit , | recur_digit , recur_digit
@abstract
@dataclass
class MatchTimes(ElementaryRE):
    pass


@abstract
@dataclass
class RecurDigit(MatchTimes):
    pass


@dataclass
class RecurDigitSingle(RecurDigit):
    digit: Annotated[str, VarRange(list(string.digits))]

    def __str__(self):
        return str(self.digit)


@dataclass
class RecurDigitMultiple(RecurDigit):
    digit: Annotated[str, VarRange(list(string.digits))]
    recur_digit: RecurDigit

    def __str__(self):
        return f"{self.digit}{self.recur_digit}"


# match_times ::= recur_digit | recur_digit ,
@dataclass
class MatchTimesSingleRecur(MatchTimes):
    recur_digit: RecurDigit
    option: Annotated[str, VarRange(["", ","])]

    def __str__(self):
        return f"{self.recur_digit}{self.option}"


# match_times ::= recur_digit ,  recur_digit
@dataclass
class MatchTimesDoubleRecur(MatchTimes):
    recur_digit1: RecurDigit
    recur_digit2: RecurDigit

    def __str__(self):
        return f"{self.recur_digit1},{self.recur_digit2}"
