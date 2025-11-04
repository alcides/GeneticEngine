# Synthetic problem generation

Synthetic problem generation lets you quickly produce random grammars and corresponding target individuals to benchmark search behavior without hand-crafting domain grammars.

This is useful for stress-testing operators, budgets, and configuration choices under controlled complexity.

## API

```{eval-rst}
.. autofunction:: geneticengine.grammar.synthetic_grammar.create_arbitrary_grammar
```

### Parameters (summary)
- **seed**: Controls reproducibility of the generated grammar.
- **non_terminals_count**: Number of abstract non-terminals to create.
- **recursive_non_terminals_count**: How many of the last non-terminals are allowed to be recursive.
- **productions_per_non_terminal(rd)**: Callable returning how many productions each non-terminal has (called per non-terminal with a `random.Random`).
- **non_terminals_per_production(rd)**: Callable returning the arity (number of fields) per production (called per production with a `random.Random`).
- **base_types**: Set of terminal/base field types to allow as leaves (defaults to `{int, bool}`).

The function returns `(nodes, root)` where `nodes` are the dynamically created classes (non-terminals and productions) and `root` is the designated root non-terminal.

## Minimal example

```python
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.synthetic_grammar import create_arbitrary_grammar
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation

# 1) Generate a random grammar
nodes, root = create_arbitrary_grammar(
    seed=0,
    non_terminals_count=3,
    recursive_non_terminals_count=2,
    productions_per_non_terminal=lambda rd: 2,
    non_terminals_per_production=lambda rd: 1,
)

# 2) Build a Grammar object
G = extract_grammar(nodes, root)

# 3) Create a random target individual (phenotype) from this grammar
r = NativeRandomSource(0)
rep = TreeBasedRepresentation(G, decider=MaxDepthDecider(r, G, G.get_min_tree_depth()))
G_target = rep.create_genotype(r, depth=8)
P_target = rep.genotype_to_phenotype(G_target)
print(P_target)
```

## Using with Genetic Programming

A common pattern is to generate a target individual and then define a distance-based fitness to measure how close candidates are to that target. For example, using string distance over the stringified phenotypes:

```python
from polyleven import levenshtein
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.problems import SingleObjectiveProblem

# Assume G and P_target are created as above

def fitness_function(p):
    return levenshtein(str(p), str(P_target))

problem = SingleObjectiveProblem(
    fitness_function=fitness_function,
    minimize=True,
    target=0,
)

r = NativeRandomSource(0)
alg = GeneticProgramming(
    problem=problem,
    budget=EvaluationBudget(100),
    representation=TreeBasedRepresentation(G, decider=MaxDepthDecider(r, G, G.get_min_tree_depth() + 10)),
    population_size=10,
    random=r,
)
ind = alg.search()[0]
print(ind, ind.get_fitness(problem))
```

## Complexity control tips
- **Grammar size**: Increase `non_terminals_count` and/or `productions_per_non_terminal` to grow the search space.
- **Depth/recursion**: Control recursion potential with `recursive_non_terminals_count`, and control tree growth with `MaxDepthDecider` and the depth you pass to `create_genotype`.
- **Arity**: Use `non_terminals_per_production` to increase/decrease branching factor per production.
- **Leaf variety**: Adjust `base_types` to change the terminal set (e.g., `{int, bool, float}`).

## End-to-end runnable example
See `examples/synthetic_grammar_example.py` for a complete script with CLI flags to vary grammar size, recursion, and arity.

