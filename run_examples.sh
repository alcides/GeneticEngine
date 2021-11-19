#!/bin/bash
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

PYTHON_BINARY=python3



function run_example {
    printf "Running $1..."
    $PYTHON_BINARY $1 > /dev/null && echo "(done)" || echo "(failed)"
    
}

# Should be somewhere else (maybe add to unit tests)
run_example examples/simple_choice_of_choice.py

run_example examples/example.py
run_example examples/pymax.py
run_example examples/vectorialgp_example.py
run_example examples/regression_example.py
run_example examples/santafe.py
run_example examples/string_match.py
run_example examples/progsys/Number_IO.py

