from __future__ import annotations

import re
import string
from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.core.decorators import abstract
from geneticengine.metahandlers.vars import VarRange


# Auxiliary lists of letters
lower_vowel = ["a", "e", "i", "o", "u"]
upper_vowel = ["A", "E", "I", "O", "U"]
lower_consonant = list(re.sub("a|e|i|o|u", "", string.ascii_lowercase))
upper_consonant = list(re.sub("A|E|I|O|U", "", string.ascii_uppercase))


# string :: letter | letter string
@dataclass
class String(ABC):
    pass


# letter ::= char | vowel | consonant
@abstract
@dataclass
class Letter(String):
    pass


# letter string
@dataclass
class LetterString(String):
    letter: Letter
    string: String

    def __str__(self):
        return str(self.letter) + str(self.string)


# char ::= " " | ! | ? | , | .
@dataclass
class Char(Letter):
    value: Annotated[str, VarRange([" ", "!", "?", ",", "."])]

    def __str__(self):
        return self.value


# vowel ::= lower_vowel | upper_vowel
# lower_vowel = a | e | i | o | u
# upper_vowel = A | E | I | O | U
@dataclass
class Vowel(Letter):
    vowel: Annotated[str, VarRange(lower_vowel + upper_vowel)]

    def __str__(self):
        return self.vowel


# consonant ::= lower_consonant | upper_consonant
# lower_consonant ::= b | c | d | f | g | h | j | k | l | ...
# upper_consonant ::= B | C | D | F | G | H | J | K | L | ...
@dataclass
class Consonant(Letter):
    consonant: Annotated[str, VarRange(lower_consonant + upper_consonant)]

    def __str__(self):
        return self.consonant
