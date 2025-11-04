#!/bin/bash
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

# Preferred Python runner: uv (falls back to system python3)
if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run -q -- python)
else
    PYTHON_CMD=(python3)
fi

# Optional: attempt to use Codon when requested, but fall back for packages Codon doesn't support
# Enable with: CODON=1 ./run_examples.sh
function can_run_with_codon {
    # Return 0 (true) if the example likely works with Codon; 1 otherwise
    local file="$1"
    # Heuristic: Codon generally doesn't support heavy third-party modules
    if grep -E '^(from|import)\s+(pandas|numpy|sklearn|seaborn|z3|pathos|matplotlib|sympy)\b' "$file" >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

cd "$(dirname "$0")"

function run_example {
    printf "Running $1..."
    if [[ "${CODON-0}" == "1" ]] && command -v codon >/dev/null 2>&1 && can_run_with_codon "$1"; then
        codon run -release "$1" > /dev/null || { echo "(failed)"; exit 111; }
        echo "(success)"
        return
    fi

    "${PYTHON_CMD[@]}" "$1" > /dev/null
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "(success)"
    else
        echo "(failed)"
        exit 111
    fi

}

# Should be somewhere else (maybe add to unit tests)
# run_example examples/simple_choice_of_choice.py

run_example examples/geml/classifier_example.py
run_example examples/geml/regressor_example.py

run_example examples/benchmarks/classification.py
run_example examples/benchmarks/classification_lexicase.py
run_example examples/benchmarks/domino.py
run_example examples/benchmarks/game_of_life_vectorial.py
run_example examples/benchmarks/mario_level.py
run_example examples/benchmarks/pymax.py
run_example examples/benchmarks/regression.py
run_example examples/benchmarks/regression_lexicase.py
run_example examples/benchmarks/santafe.py
run_example examples/benchmarks/string_match.py
run_example examples/benchmarks/vectorialgp.py


run_example examples/progsys/Number_IO.py
run_example examples/progsys/Median.py
run_example examples/progsys/Smallest.py
run_example examples/progsys/Sum_of_Squares.py


# Other features

run_example examples/example.py
run_example examples/classification_unknown_length_objectives.py
run_example examples/binary.py
run_example examples/multi_target_lexicase.py
run_example examples/pcfg_example.py
run_example examples/recurrence.py
run_example examples/synthetic_grammar_example.py
run_example examples/tutorial_example.py
run_example examples/logging_example.py
