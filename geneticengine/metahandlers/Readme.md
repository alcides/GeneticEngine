Metahandlers
============

Motivation
----------

GGGP solution spaces often include base types that are not easily represented in BNF syntax. For example, integers are often represented as follows:
```
<I> ::= <P>
      | -<P>
<P> ::= <D><P>
<D> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 9
```
Not only is this cumbersome, this can result to issues (like integer overflowing).  In our STGP front-end approach, the user does not need to deal with this issue, as only the base type needs to be specified, and Genetic Engine handles the rest.

However, further restriction of the base types can be useful (maybe we only want positive integers/even numbers or floats between 0 and 1, in the case of probabilities). To allow the user this flexibility in Genetic Engine, we introduce metahandlers, inspired by [Refinement Types GP](https://link.springer.com/chapter/10.1007/978-3-030-58115-2_2). Metahandlers also allow the user to implement specific mutation and crossover methods for each base type in the grammar.

Here we only discuss the usage of metahandlers. For implementation details we refer to section 3.2 in [this paper](https://arxiv.org/abs/2210.04826).

Usage
-----

We have implemented a number of standard metahandlers, like [IntRange, IntList](ints.py), [FloatRange, FloatList](floats.py), [VarRange](vars.py) and [ListSizeBetween](lists.py). Please see the respective files for the usage of these metahandlers. In the [running example](../../), the IntRange is already introduced. In the [examples](../../examples/), other metahandlers are used, and can there be consulted for inspiration.

Create your own metahandler
---------------------------

We have intended for metahandlers to be easily implemented by the user. Suggestions on improving this are very welcome.

Metahandlers should follow the [MetaHandlerGenerator](base.py) protocol. More specifically, it must have a generator method, which specifies how a node with the metahandler type should be generated. Consult our predefined metahandlers for inspiration.

Adding type specific mutations and crossover
--------------------------------------------

For some base types, the user might want to include type specific mutations and crossovers. For example, in the GP-based feature learning method [M3GP](https://link.springer.com/chapter/10.1007/978-3-319-16501-1_7), list-specific mutation and crossover methods are introduced. We have introduced those methods in the [ListSizeBetween](lists.py) metahandler.

To introduce a type-specific mutation or crossover, one should create a metahandler with a mutation or/and crossover method. The methods should take in the same parameters as the methods in [ListSizeBetween](lists.py), and they should return a single individual of the same base type the metahandler defines.
