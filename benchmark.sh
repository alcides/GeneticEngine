export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

PYTHON_BINARY=python3
SEED=0

mkdir -p benchmark_run


#$PYTHON_BINARY examples/classification.py --crossover 0.75 --mutation 0.01 --max_depth 15 --generations 50 --population_size 500 --elites 5 --csv benchmark_run/classification.csv --seed $SEED
#$PYTHON_BINARY examples/game_of_life.py --crossover 0.75 --mutation 0.01 --max_depth 15 --generations 50 --population_size 200 --elites 5 --csv benchmark_run/game_of_live.csv --seed $SEED
$PYTHON_BINARY examples/game_of_life_vectorial.py --crossover 0.75 --mutation 0.01 --max_depth 15 --generations 50 --population_size 200 --elites 5 --csv benchmark_run/game_of_live_vectorial.csv --seed $SEED

# $PYTHON_BINARY examples/example.py --population_size 150 --generations 40 --max_depth 5
