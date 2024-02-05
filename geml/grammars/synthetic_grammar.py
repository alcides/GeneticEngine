from __future__ import annotations

import string
from abc import ABC
from dataclasses import dataclass
from random import Random
from typing import Any
from typing import Callable
from typing import TypeVar

from geneticengine.core.decorators import abstract


def create_dataclass_dynamically(
    name: str,
    args: dict[str, Any] = {},
    annotations: dict[str, Any] = {},
    parent_class: type = ABC,
) -> type:
    """Dynamically creates a new data class with the given name and arguments.

    Args:
        name (str): the name of the new data class
        args (dict[str, Any]): the attributes and values for the new data class. Default is an empty dictionary.
        annotations (dict[str, Any]): the type annotations for the new data class. Default is an empty dictionary.
        parent_class (type): the parent class for the new data class. Default is ABC (abstract base class).

    Returns:
        The new data class created
    """
    new_data_class = type(name, (parent_class,), args)

    if annotations:
        if not hasattr(new_data_class, "__annotations__"):
            new_data_class.__annotations__ = {}

        for key in annotations:
            new_data_class.__annotations__[key] = annotations[key]

    return dataclass(new_data_class)


def make_non_terminal_names(count: int):
    """Returns a list of names for non_terminal_symbols."""
    return [f"NT_{i:04}" for i in range(count)]


def make_non_terminal_class(name: str):
    """Returns a new abstract class, with the provided name."""
    return abstract(type(name, (ABC,), {}))


def make_production(index: int, nt: type, field_types: list[type]) -> type:
    """Creates a class representing a Production."""
    name = f"{nt.__name__}_P_{index:05}"
    annotations = {f"{i:_>3}": ft for (i, ft) in zip(string.ascii_lowercase, field_types)}
    return dataclass(create_dataclass_dynamically(name, {}, annotations, nt))


A = TypeVar("A")


def select_random(rd: Random, how_many: int, candidates: list[A]) -> set[A]:
    return {rd.choice(candidates) for _ in range(how_many)}


def create_arbitrary_grammar(
    seed: int,
    non_terminals_count: int,
    recursive_non_terminals_count: int,
    productions_per_non_terminal: Callable[[Random], int] = lambda rd: round(
        rd.uniform(1, 10),
    ),
    non_terminals_per_production: Callable[[Random], int] = lambda rd: round(
        rd.uniform(0, 10),
    ),
    base_types: set[type] = {int, bool},
) -> tuple[list[type], type]:
    """Generates a random grammar, based on a particular seed."""
    rd = Random(seed)

    non_terminal_names = make_non_terminal_names(non_terminals_count)
    non_terminals = [make_non_terminal_class(name) for name in non_terminal_names]
    productions = []

    field_candidates = list(base_types)
    for (index, nt) in enumerate(non_terminals):
        allow_recursion = index > (non_terminals_count - recursive_non_terminals_count)

        for production_index in range(productions_per_non_terminal(rd)):
            # At least one production should not be recursive
            recursion_in_production = allow_recursion and production_index > 0

            candidates = field_candidates if recursion_in_production else field_candidates + [nt]

            annotations: list[type] = [rd.choice(candidates) for _ in range(non_terminals_per_production(rd))]
            prod = make_production(production_index, nt, annotations)
            productions.append(prod)
        field_candidates.append(nt)

    return (productions + non_terminals, non_terminals[-1])
