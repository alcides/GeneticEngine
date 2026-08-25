"""Parse PonyGE2 BNF grammars into GeneticEngine grammar classes."""

from __future__ import annotations

import ast
from dataclasses import dataclass, make_dataclass
from pathlib import Path
import re
from typing import Annotated, Any, Mapping

from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.vars import VarRange


class PonyGE2GrammarError(ValueError):
    """Raised when a PonyGE2 grammar cannot be converted."""


@dataclass(frozen=True)
class _Rule:
    name: str
    alternatives: list[str]


@dataclass
class PonyGE2Grammar:
    """The generated GeneticEngine structure for a PonyGE2 grammar."""

    grammar: Grammar
    nodes: list[type]
    starting_symbol: type
    nonterminals: dict[str, type]
    productions: dict[str, list[type]]

    def to_geneticengine(self) -> tuple[list[type], type]:
        """Return the generated node classes and starting symbol."""
        return self.nodes, self.starting_symbol


_RULE = re.compile(r"<(?P<name>[^<>\s]+)>\s*::=")
_NONTERMINAL = re.compile(r"^<(?P<name>[^<>\s]+)>$")


def _remove_comments(source: str) -> str:
    result: list[str] = []
    quote: str | None = None
    escaped = False
    comment = False
    for char in source:
        if comment:
            if char == "\n":
                comment = False
                result.append(char)
            continue
        if escaped:
            result.append(char)
            escaped = False
        elif quote is not None and char == "\\":
            result.append(char)
            escaped = True
        elif char in "'\"":
            quote = None if quote == char else char if quote is None else quote
            result.append(char)
        elif char == "#" and quote is None:
            comment = True
        else:
            result.append(char)
    return "".join(result)


def _split_alternatives(body: str) -> list[str]:
    alternatives: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(body):
        if escaped:
            escaped = False
        elif quote is not None and char == "\\":
            escaped = True
        elif char in "'\"":
            quote = None if quote == char else char if quote is None else quote
        elif char == "|" and quote is None:
            alternatives.append(body[start:index].strip())
            start = index + 1
    alternatives.append(body[start:].strip())
    return alternatives


def _parse_rules(source: str) -> list[_Rule]:
    source = _remove_comments(source)
    matches = list(_RULE.finditer(source))
    if not matches:
        raise PonyGE2GrammarError("No PonyGE2 rules were found; expected '<name> ::= ...'.")

    rules: list[_Rule] = []
    for index, match in enumerate(matches):
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        body = source[match.end() : body_end].strip()
        name = f"<{match.group('name')}>"
        if not body:
            raise PonyGE2GrammarError(f"Rule {name} has no production alternatives.")
        rules.append(_Rule(name, _split_alternatives(body)))

    names = [rule.name for rule in rules]
    if len(names) != len(set(names)):
        raise PonyGE2GrammarError("Each nonterminal must be defined exactly once.")
    return rules


def _read_source(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text()
    if "::=" not in source and Path(source).is_file():
        return Path(source).read_text()
    return source


def _quoted_terminal(token: str) -> str:
    try:
        value = ast.literal_eval(token)
    except (SyntaxError, ValueError) as error:
        raise PonyGE2GrammarError(f"Invalid quoted terminal {token!r}.") from error
    if not isinstance(value, str):
        raise PonyGE2GrammarError(f"Quoted terminal {token!r} is not a string.")
    return value


def _tokenize_alternative(alternative: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    index = 0
    while index < len(alternative):
        if alternative[index] == "<":
            end = alternative.find(">", index + 1)
            if end < 0:
                raise PonyGE2GrammarError(f"Unclosed nonterminal in {alternative!r}.")
            symbol = alternative[index : end + 1]
            if not _NONTERMINAL.fullmatch(symbol):
                raise PonyGE2GrammarError(f"Invalid nonterminal {symbol!r}.")
            tokens.append(("nonterminal", symbol))
            index = end + 1
            continue

        if alternative[index] in "'\"":
            quote = alternative[index]
            end = index + 1
            escaped = False
            while end < len(alternative):
                if not escaped and alternative[end] == quote:
                    break
                escaped = not escaped and alternative[end] == "\\"
                if alternative[end] != "\\":
                    escaped = False
                end += 1
            if end >= len(alternative):
                raise PonyGE2GrammarError(f"Unclosed quoted terminal in {alternative!r}.")
            tokens.append(("terminal", _quoted_terminal(alternative[index : end + 1])))
            index = end + 1
            continue

        end = index
        while end < len(alternative) and alternative[end] not in "<'\"":
            end += 1
        text = alternative[index:end]
        if text.strip():
            tokens.append(("terminal", text))
        index = end
    return tokens


def _class_name(symbol: str, used: set[str]) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", symbol)
    name = "".join(part[:1].upper() + part[1:] for part in parts) or "Symbol"
    if name[0].isdigit():
        name = f"N{name}"
    candidate = name
    suffix = 2
    while candidate in used:
        candidate = f"{name}{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _terminal_type(value: str) -> Any:
    return Annotated[str, VarRange([value])]


def _range_type(specification: str, constants: Mapping[str, int]) -> Any:
    name = specification.removeprefix("GE_RANGE:")
    try:
        size = int(name)
    except ValueError:
        if name not in constants:
            raise PonyGE2GrammarError(
                f"GE_RANGE:{name} needs an integer or a value in ge_constants.",
            )
        size = constants[name]
    if size <= 0:
        raise PonyGE2GrammarError(f"GE_RANGE must have a positive size, got {size}.")
    return Annotated[int, IntRange(0, size - 1)]


def parse_ponyge2_grammar(
    source: str | Path,
    *,
    ge_constants: Mapping[str, int] | None = None,
) -> PonyGE2Grammar:
    """Parse PonyGE2 BNF text or a grammar file.

    The first rule is the starting symbol. Ordinary terminals become fixed
    ``VarRange`` string fields. ``GE_RANGE:n`` becomes an integer field in
    ``range(n)``; named ranges can be supplied through ``ge_constants``.
    """
    constants = ge_constants or {}
    rules = _parse_rules(_read_source(source))
    rule_names = {rule.name for rule in rules}

    used_names: set[str] = set()
    nonterminals = {
        rule.name: abstract(type(_class_name(rule.name, used_names), (object,), {"__module__": __name__}))
        for rule in rules
    }

    productions: dict[str, list[type]] = {}
    nodes: list[type] = list(nonterminals.values())
    for rule in rules:
        generated: list[type] = []
        for index, alternative in enumerate(rule.alternatives):
            if alternative.startswith("GE_"):
                if not alternative.startswith("GE_RANGE:"):
                    raise PonyGE2GrammarError(f"Unsupported PonyGE2 special constant {alternative!r}.")
                field_types = [_range_type(alternative, constants)]
            else:
                field_types = []
                for kind, value in _tokenize_alternative(alternative):
                    if kind == "nonterminal":
                        if value not in rule_names:
                            raise PonyGE2GrammarError(f"Undefined nonterminal {value}.")
                        field_types.append(nonterminals[value])
                    else:
                        field_types.append(_terminal_type(value))

            annotations = {f"field_{field_index}": field_type for field_index, field_type in enumerate(field_types)}
            production_name = f"{nonterminals[rule.name].__name__}P{index}"
            production = make_dataclass(
                production_name,
                [(field_name, field_type) for field_name, field_type in annotations.items()],
                bases=(nonterminals[rule.name],),
                namespace={"__module__": __name__},
            )
            generated.append(production)
            nodes.append(production)
        productions[rule.name] = generated

    starting_symbol = nonterminals[rules[0].name]
    grammar = extract_grammar(nodes, starting_symbol)
    return PonyGE2Grammar(grammar, nodes, starting_symbol, nonterminals, productions)


def load_ponyge2_grammar(path: str | Path, *, ge_constants: Mapping[str, int] | None = None) -> PonyGE2Grammar:
    """Parse a PonyGE2 grammar file."""
    return parse_ponyge2_grammar(Path(path), ge_constants=ge_constants)
