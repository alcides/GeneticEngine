#!/bin/bash
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

PYTHON_BINARY=python3

set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

cd "$(dirname "$0")"

function run_example {
    printf "Running $1..."
    $PYTHON_BINARY $1 > /dev/null
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "(success)"
    else
        echo "(failed)"
        exit 111
    fi

}

# # Should be somewhere else (maybe add to unit tests)
# run_example examples/simple_choice_of_choice.py

run_example examples/example.py
run_example examples/pymax.py
run_example examples/vectorialgp_example.py
run_example examples/regression.py
run_example examples/regression_lexicase.py
run_example examples/classification.py
run_example examples/classification_probabilisticGE.py
run_example examples/classification_lexicase.py
run_example examples/sklearn_type_examples.py
run_example examples/santafe.py
run_example examples/game_of_life.py
run_example examples/string_match.py

run_example examples/progsys/Number_IO.py
run_example examples/progsys/Median.py
run_example examples/progsys/Smallest.py
run_example examples/progsys/Sum_of_Squares.py

# Other features

run_example examples/binary.py
run_example examples/exponentiation.py
run_example examples/game_of_life_vectorial.py
run_example examples/multi_target_lexicase.py
run_example examples/multipopulation.py
run_example examples/pcfg_example.py
run_example examples/recurrence.py
run_example examples/synthetic_grammar_example.py
run_example examples/tutorial_example.py
