from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Annotated, Any, Optional, Union

import numpy as np
from numpy import dtype
import pandas as pd


from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.grammar.metahandlers.vars import VarRange
from geneticengine.problems import MultiObjectiveProblem, Problem


def read_edge_and_random(root_dir: Union[str, Path], dataset: str) -> pd.DataFrame:
    edge_filename = os.path.join(root_dir, f"{dataset}/{dataset}-edge.json")
    random_filename = os.path.join(root_dir, f"{dataset}/{dataset}-random.json")
    edge = pd.read_json(edge_filename, lines=True).assign(edge_case=True)
    random = pd.read_json(random_filename, lines=True).assign(edge_case=False)
    return pd.concat([edge, random])


def read_dataset(dataset, cache_dir) -> pd.DataFrame:
    df: Optional[pd.DataFrame] = None
    if cache_dir:
        df = read_edge_and_random(cache_dir, dataset)
    else:
        raise Exception("Did not provide the path to the PSB dataset")
    if dataset == "replace-space-with-newline":
        df = df.rename(columns={"output1": "stdout", "output2": "output"})
    else:
        df = df.rename(columns={"output1": "output", "output2": "stdout"})
    return df


def convert_dtype_to_stack(t: dtype) -> str:
    if t == np.int64:
        return "ints"
    elif t == object:
        return "strings"
    else:
        raise Exception(f"{t} does not have a stack")


defaults = {
    "ints": 0,
    "bools": 0,
    "floats": 0.0,
    "strings": "",
}


class Instruction(ABC): ...


@dataclass
class Program:
    instructions: Annotated[list[Instruction], ListSizeBetween(1, 500)]


@dataclass
class Swap(Instruction):
    stack: Annotated[str, VarRange(list(defaults.keys()))]


@dataclass
class Dup(Instruction):
    stack: Annotated[str, VarRange(list(defaults.keys()))]


@dataclass
class Rotate(Instruction):
    stack: Annotated[str, VarRange(list(defaults.keys()))]


@dataclass
class BooleanLiteral(Instruction):
    val: bool


@dataclass
class BooleanAnd(Instruction):
    pass


@dataclass
class BooleanOr(Instruction):
    pass


@dataclass
class BooleanNot(Instruction):
    pass


@dataclass
class BooleanXor(Instruction):
    pass


@dataclass
class BooleanIsZero(Instruction):
    pass


@dataclass
class IntLiteral(Instruction):
    val: int


@dataclass
class IntEq(Instruction):
    pass


@dataclass
class IntLt(Instruction):
    pass


lang_operators = [
    Swap,
    Dup,
    Rotate,
    BooleanLiteral,
    BooleanAnd,
    BooleanOr,
    BooleanNot,
    BooleanXor,
    BooleanIsZero,
    IntLiteral,
    IntEq,
    IntLt,
]


def pop(stacks, type):
    if stacks[type]:
        return stacks[type].pop()
    else:
        return defaults[type]


def program_eval(
    e: Program,
    row,
    input_columns: list[tuple[str, dtype]],
    output_columns: list[tuple[str, dtype]],
) -> tuple[str, Any]:
    stdout: list[str] = []
    stacks: dict[str, list[Any]] = {
        "ints": [],
        "bools": [],
        "floats": [],
        "strings": [],
    }

    for col, ty in input_columns:
        stacks[convert_dtype_to_stack(ty)].append(getattr(row, col))

    for _, ins in zip(e.instructions, range(500)):
        try:
            match ins:
                case Swap(stack=s):
                    a = stacks[s].pop()
                    stacks[s].insert(1, a)
                case Dup(stack=s):
                    stacks[s].insert(0, stacks[s][0])
                case Rotate(stack=s):
                    stacks[s].append(stacks[s].pop())
                case BooleanAnd():
                    a = stacks["bools"].pop()
                    b = stacks["bools"].pop()
                    stacks["bools"].insert(0, a and b)
                case BooleanOr():
                    a = stacks["bools"].pop()
                    b = stacks["bools"].pop()
                    stacks["bools"].insert(0, a or b)
                case BooleanNot():
                    a = stacks["bools"].pop()
                    stacks["bools"].insert(0, not a)
                case BooleanXor():
                    a = stacks["bools"].pop()
                    b = stacks["bools"].pop()
                    stacks["bools"].insert(0, a ^ b)
                case BooleanIsZero():
                    stacks["bools"].insert(0, stacks["ints"][0] == 0)
                case BooleanLiteral(val=v):
                    stacks["bools"].insert(0, v)
                case IntLiteral(val=v):
                    stacks["ints"].insert(0, v)
                case IntEq():
                    a = stacks["ints"].pop()
                    b = stacks["ints"].pop()
                    stacks["bools"].insert(0, a == b)
                case IntLt():
                    a = stacks["ints"].pop()
                    b = stacks["ints"].pop()
                    stacks["bools"].insert(0, a < b)
                case _:
                    print("failed to process", ins)
        except Exception as e:
            pass

    return ("".join(stdout), [pop(stacks, convert_dtype_to_stack(ty)) for (name, ty) in output_columns])


def load_dataframe(program: str, path_to_psb1: str):
    df = read_dataset(program, cache_dir=path_to_psb1)
    del df["edge_case"]
    return df


class PSBBenchmark(Benchmark):
    def __init__(self, program: str, path_to_psb1: str = "datasets"):
        df = load_dataframe(program, path_to_psb1).head(10)
        self.setup_problem(df)
        self.setup_grammar(df)

    def setup_problem(self, df):

        # Problem
        def fitness_function(p: Program) -> list[Any]:
            results = []
            input_data = df
            if "stdout" in df.columns:
                stdout = df["stdout"].values
                input_data = input_data.drop(["stdout"], axis="columns")
            else:
                stdout = ["" for _ in range(input_data.shape[0])]

            output_columns = [
                (col_name, input_data.dtypes[col_name])
                for col_name in ["output", "output1", "output2"]
                if col_name in input_data.columns
            ]
            output_vals = [input_data[col_name].values for (col_name, _) in output_columns]
            input_data = input_data.drop([name for (name, _) in output_columns], axis="columns")
            col_info = list(zip(input_data.columns, input_data.dtypes))

            for row, stdout_expected, *outputs_expected in zip(
                input_data.itertuples(index=False),
                stdout,
                *output_vals,
            ):
                stdout, outputs = program_eval(p, row, col_info, output_columns)
                score_stdout = 0 if stdout == stdout_expected else 1
                score_output = sum([0 if o == eo else 1 for (o, eo) in zip(outputs, outputs_expected)])
                score = score_output + score_stdout
                results.append(score)
            return results

        self.problem = MultiObjectiveProblem(
            minimize=[True for _ in range(df.shape[0])],
            fitness_function=fitness_function,
            target=[0 for _ in range(df.shape[0])],
        )

    def setup_grammar(self, df):
        self.grammar = extract_grammar(lang_operators, Program)

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    path_to_psb1 = "/Users/alcides/Downloads/program-synthesis-benchmark-datasets-master/datasets/"
    example_run(PSBBenchmark(program="smallest", path_to_psb1=path_to_psb1))
