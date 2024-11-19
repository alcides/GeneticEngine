from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, make_dataclass
import os
from pathlib import Path
from typing import Annotated, Any, Optional, Union

import numpy as np
import pandas as pd
from difflib import SequenceMatcher


from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.ints import IntRange
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


class Int(ABC):

    def eval(self): ...


class Bool(ABC):

    def eval(self): ...


class String(ABC):

    def eval(self): ...


@dataclass
class IntToString(String):
    i: Int

    def eval(self):
        return [f"{v}" for v in self.i.eval()]


@dataclass
class AppendChar(String):
    s: String
    c: Annotated[int, IntRange(0, 128)]

    def eval(self):
        return [f"{s}{chr(self.c)}" for s in self.s.eval()]


@dataclass
class BoolIf(Int):
    c: Bool
    a: Bool
    b: Bool

    def eval(self):
        return np.where(self.c.eval(), self.a.eval(), self.b.eval())


@dataclass
class IntLt(Bool):
    a: Int
    b: Int

    def eval(self):
        return self.a.eval() < self.b.eval()


@dataclass
class IntEq(Bool):
    a: Int
    b: Int

    def eval(self):
        return self.a.eval() == self.b.eval()


@dataclass
class IntIf(Int):
    c: Bool
    a: Int
    b: Int

    def eval(self):
        return np.where(self.c.eval(), self.a.eval(), self.b.eval())


def load_dataframe(program: str, path_to_psb1: str):
    df = read_dataset(program, cache_dir=path_to_psb1)
    del df["edge_case"]
    return df


def prepare_input_and_outputs(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, np.dtype]], str, list[tuple[str, np.dtype]], pd.DataFrame]:
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
    output_vals = pd.DataFrame({col_name: input_data[col_name].values for (col_name, _) in output_columns})

    input_data = input_data.drop([name for (name, _) in output_columns], axis="columns")
    input_columns = list(zip(input_data.columns, input_data.dtypes))

    return input_data, input_columns, stdout, output_columns, output_vals


def sequence_matcher_edits(word_a, word_b):
    required_edits = [
        code for code in (SequenceMatcher(a=word_a, b=word_b, autojunk=False).get_opcodes()) if code[0] != "equal"
    ]
    return len(required_edits)


def diff(ty, series_real, series_expected):
    if ty == np.int64:
        return series_real - series_expected
    elif ty == np.bool:
        return series_real - series_expected
    elif ty == np.object_:
        return np.array([sequence_matcher_edits(a, b) for (a, b) in zip(series_real, series_expected)])
    else:
        raise Exception(f"{ty} not supported")


def convert(ty):
    if ty == np.int64:
        return Int
    elif ty == np.bool:
        return Bool
    elif ty == np.object_:
        return String
    else:
        raise Exception(f"{ty} not supported")


class PSBBenchmark(Benchmark):
    def __init__(self, program: str, path_to_psb1: str = "datasets"):
        df = load_dataframe(program, path_to_psb1).head(10)
        input_data, input_columns, stdout, output_columns, output_vals = prepare_input_and_outputs(df)
        self.setup_problem(input_data, output_columns, output_vals)
        self.setup_grammar(input_columns, input_data, output_columns)

    def setup_problem(self, input_data, output_columns, output_vals: pd.DataFrame):

        # Problem
        def fitness_function(p: Any) -> list[Any]:

            output = p.eval()
            for k in output:
                output[k] = np.reshape(output[k], shape=(output_vals.shape[0],))

            full_results = pd.DataFrame({col: diff(ty, output[col], output_vals[col]) for (col, ty) in output_columns})
            return full_results.sum(axis=1)

        self.problem = MultiObjectiveProblem(
            minimize=[True for _ in range(input_data.shape[0])],
            fitness_function=fitness_function,
            target=[0 for _ in range(input_data.shape[0])],
        )

    def setup_grammar(self, input_columns, input_data, output_columns):
        Program = make_dataclass("Program", [(name, convert(ty)) for name, ty in output_columns])

        def f(self):
            return {name: getattr(self, name).eval() for name, _ in output_columns}

        custom_inputs = []
        for col, ty in input_columns:
            ctype = convert(ty)
            var = make_dataclass(col, fields=[], bases=(ctype,))

            def k(self):
                return input_data[col]

            var.eval = k

            import inspect

            print(inspect.isabstract(var))
            custom_inputs.append(var)

        Program.eval = f
        n_instances = input_data.shape[0]

        @dataclass
        class BoolLit(Bool):
            val: bool

            def eval(self):
                return np.full((n_instances,), self.val)

        @dataclass
        class IntLit(Int):
            val: Annotated[int, VarRange([x for x in range(-256, 256)])]

            def eval(self):
                return np.full((n_instances,), self.val)

        @dataclass
        class EmptyString(String):

            def eval(self):
                return np.full((n_instances,), "")

        self.grammar = extract_grammar(
            [
                Bool,
                Int,
                String,
                BoolLit,
                BoolIf,
                IntLit,
                IntIf,
                IntLt,
                IntEq,
                EmptyString,
                AppendChar,
                IntToString,
            ]
            + custom_inputs,
            Program,
        )

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    path_to_psb1 = "/Users/alcides/Downloads/program-synthesis-benchmark-datasets-master/datasets/"
    example_run(PSBBenchmark(program="checksum", path_to_psb1=path_to_psb1))
