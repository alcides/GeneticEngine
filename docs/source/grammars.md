# Grammars


## Defining grammars

The [tutorial](tutorial.md) contains an example of how to create a grammar.

The examples folder in the repo also contain several examples of grammars, including examples for classification and regressions, program synthesis and string matching.


## Using builtin grammar definitions

The `geneticengine.grammars` module contains many different ready implementations of grammars popular in Genetic Programming:

* Standard Genetic Programming
* Literals
* Regex
* Basic Math
* Coding
* Letters

## Controlling the Depth of Individuals

For backwards compatibility with [PonyGE2](https://github.com/PonyGE/PonyGE2), another Grammar-Guided Genetic Programming Framework, GeneticEngine supports two modes of defining the depth of individuals.

### Depth of the tree

```python
grammar = extract_grammar([A, B], Root, expansion_depthing=False)
```

Using this method, the maximum depth of an individual is the depth of its tree representation.

### Depth of the grammatical expansion

```python
grammar = extract_grammar([A, B], Root, expansion_depthing=True)
```

We also support grammar-expansion depthing, as done in [PonyGE2](https://github.com/PonyGE/PonyGE2). In grammar-expansion depthing the depth is increased each time a grammar production rule is expanded. For example, suppose you have the following grammar:

```
A       :== 0 | B
B       :== 1 | 2
```

A tree with a single node of value 0 will have a depth of 2, as the expansion is A -> 0, where node A has depth 1, and node 0 will have a depth of 2. A tree with a single node of value 1 (or 2) derives from the expansion A -> B -> 1 (or 2), and will thus have a depth of 3, even though it consists of only a single node.

### On the difference of both approaches

In Genetic Engine we don't keep track of the depth of nodes, but only of the distance to (the deepest) terminal and number of nodes (consultable by applying the gengy_distance_to_term and gengy_nodes methods to any tree node). When using grammar-expansion depthing, you can only read the distance to terminal and number of nodes of actual objects. Therefore, reading the distance to terminal of a tree with the single node 0 as introduced above, will not give you the distance to the terminal of the whole tree (which would be 2), but that of the node 0, which is 1. The algorithm works correctly, but keep this in mind at consultation.
