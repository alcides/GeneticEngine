#!/bin/bash
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

PYTHON_BINARY=python



function run_example {
    printf "Running $1..."
    $PYTHON_BINARY $1 > /dev/null && echo "(done)" || echo "(failed)"

}

# # Should be somewhere else (maybe add to unit tests)
# run_example examples/simple_choice_of_choice.py

run_example examples/example.py
run_example examples/pymax.py # Works in PonyGE
run_example examples/vectorialgp_example.py
run_example examples/regression.py # Works in PonyGE
run_example examples/classification.py # Works in PonyGE
run_example examples/santafe.py
run_example examples/game_of_life.py
run_example examples/string_match.py # Works in PonyGE

run_example examples/progsys/Number_IO.py # Works in PonyGE
run_example examples/progsys/Median.py # Works in PonyGE: PonyGE gives very bad results
run_example examples/progsys/Smallest.py # Works in PonyGE: PonyGE gives very bad results
run_example examples/progsys/Sum_of_Squares.py # Works in PonyGE: PonyGE gives very bad results
