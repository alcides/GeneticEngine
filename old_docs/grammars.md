# Grammars


## Defining grammars

The [tutorial](tutorial.md) contains an example of how to create a grammar.

The examples folder in the repo also contain several examples of grammars, including examples for classification and regressions, program synthesis and string matching.

## A note on defining Depth of trees

Depthing can be adjusted when extract the grammar, with the variable expansion_depthing (bool, default = False) in the [extract_grammar](../geneticengine/core/grammar.py) method.

We refer to the tree-depth-measurement method as the _depthing_ method. Genetic Engine supports two depthing methods. First, it supports standard abstract-syntax-tree depthing, in which the depth is increased each time you go done one node in the tree. This is straight-forward and intuitive, so we will not elaborate on this.

We also support grammar-expansion depthing, as done in [PonyGE2](https://github.com/PonyGE/PonyGE2). In grammar-expansion depthing the depth is increased each time a grammar production rule is expanded. For example, suppose you have the following grammar:

```
A       :== 0 | B
B       :== 1 | 2
```

A tree with a single node of value 0 will have a depth of 2, as the expansion is A -> 0, where node A has depth 1, and node 0 will have a depth of 2. A tree with a single node of value 1 (or 2) derives from the expansion A -> B -> 1 (or 2), and will thus have a depth of 3, even though it consists of only a single node.

In Genetic Engine we don't keep track of the depth of nodes, but only of the distance to (the deepest) terminal and number of nodes (consultable by applying the gengy_distance_to_term and gengy_nodes methods to any tree node). When using grammar-expansion depthing, you can only read the distance to terminal and number of nodes of actual objects. Therefore, reading the distance to terminal of a tree with the single node 0 as introduced above, will not give you the distance to the terminal of the whole tree (which would be 2), but that of the node 0, which is 1. The algorithm works correctly, but keep this in mind at consultation.
