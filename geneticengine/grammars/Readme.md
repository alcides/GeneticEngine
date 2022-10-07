# Grammars

In the [root folder](https://github.com/alcides/GeneticEngine) there is an example on how to create a grammar. In this folder we introduced a number of off-the-shelf grammars that can be used for many general problems, like classification and regression, program synthesis and string matching. See the [examples folder](../../examples/) for worked-out examples of these.

## Depthing
Depthing can be adjusted when extract the grammar, with the variable expansion_depthing (bool, default = False) in the [extract_grammar](../core/grammar.py) method.

We refer to the tree-depth-measurement method as the _depthing_ method. Genetic Engine supports two depthing methods. First, it supports standard abstract-syntax-tree depthing, in which the depth is increased each time you go done one node in the tree. This is straight-forward and intuitive, so we will not elaborate on this.

We also support grammar-expansion depthing, as done in [PonyGE2](https://github.com/PonyGE/PonyGE2). In grammar-expansion depthing the depth is increased each time a grammar production rule is expanded. For example, suppose you have the following grammar:

```
A       :== 0 | B
B       :== 1 | 2
```

A tree with a single node of value 0 will have a depth of 2, as the expansion is A -> 0, where node A has depth 1, and node 0 will have a depth of 2. A tree with a single node of value 1 (or 2) derives from the expansion A -> B -> 1 (or 2), and will thus have a depth of 3, even though it consists of only a single node.
