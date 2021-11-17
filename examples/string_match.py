import re
import string
from abc import ABC
from dataclasses import dataclass
from textwrap import indent
from typing import Annotated

from geneticengine.metahandlers.vars import VarRange
from geneticengine.core.grammar import extract_grammar
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.representations.treebased import treebased_representation
'''
Auxiliary lists of letters
'''
lower_vowel = ['a', 'e', 'i', 'o', 'u']
upper_vowel = ['A', 'E', 'I', 'O', 'U']
lower_consonant = list(re.sub('a|e|i|o|u', '', string.ascii_lowercase))
upper_consonant = list(re.sub('A|E|I|O|U', '', string.ascii_uppercase))


# string :: letter | letter string
@dataclass
class String(ABC):
    pass


# letter ::= char | vowel | consonant
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


# vowel ::= lower_vowl | upper_vowel
# lower_vowel = a | e | i | o | u
# upper_vowel = A | E | I | O | U
@dataclass
class Vowel(Letter):
    pass


@dataclass
class LowerVowel(Vowel):
    value: Annotated[str, VarRange(lower_vowel)]

    def __str__(self):
        return self.value


@dataclass
class UpperVowel(Vowel):
    value: Annotated[str, VarRange(upper_vowel)]

    def __str__(self):
        return self.value


# consonant ::= lower_consonant | upper_consonant
# lower_consonant ::= b | c | d | f | g | h | j | k | l | ...
# upper_consonant ::= B | C | D | F | G | H | J | K | L | ...
@dataclass
class Consonant(Letter):
    pass


@dataclass
class LowerConsonant(Consonant):
    value: Annotated[str, VarRange(lower_consonant)]

    def __str__(self):
        return self.value


@dataclass
class UpperConsonant(Consonant):
    value: Annotated[str, VarRange(upper_consonant)]

    def __str__(self):
        return self.value


def fit(individual: String):
    guess = str(individual)
    target = 'OLA'
    fitness = max(len(target), len(guess))
    # Loops as long as the shorter of two strings
    for (t_p, g_p) in zip(target, guess):
        if t_p == g_p:
            # Perfect match.
            fitness -= 1
        else:
            # Imperfect match, find ASCII distance to match.
            fitness -= 1 / (1 + (abs(ord(t_p) - ord(g_p))))
    return fitness


fitness_function = lambda x: fit(x)

if __name__ == "__main__":
    g = extract_grammar([
        LetterString, Char, LowerVowel, UpperVowel, LowerConsonant,
        UpperConsonant
    ], String)
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        max_depth=10,
        population_size=100,
        number_of_generations=100,
        minimize=True,
    )
    print("Started running...")
    (b, bf, bp) = alg.evolve(verbose=0)
    print(bp)
    print(b)
    print("---")
    print(str(bp))
    print(str(b))
    print("With fitness: {}".format(bf))
