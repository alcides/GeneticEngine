#!/bin/bash
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

PYTHON_BINARY=python3



function run_example {
    printf "Running $1..."
    $PYTHON_BINARY $1 > /dev/null && echo "(done)" || echo "(failed)"
    
}


run_example examples/example.py
run_example examples/pymax.py
run_example examples/vectorialgp_example.py
run_example examples/regression_example.py
run_example examples/santafe.py
run_example examples/progsys/Number_IO.py
run_example examples/game_of_life.py
