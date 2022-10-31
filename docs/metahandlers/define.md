# Create your own metahandler

We have intended for metahandlers to be easily implemented by the user. Suggestions on improving this are very welcome.

# Protocol

Metahandlers should follow the `MetaHandlerGenerator` protocol. More specifically, it must have a generator method, which specifies how a node with the metahandler type should be generated. Consult our predefined metahandlers for inspiration.

::: geneticengine.metahandlers.base.MetaHandlerGenerator

## Adding type specific mutations and crossover

For some base types, the user might want to include type specific mutations and crossovers. For example, in the GP-based feature learning method [M3GP](https://link.springer.com/chapter/10.1007/978-3-319-16501-1_7), list-specific mutation and crossover methods are introduced. We have introduced those methods in the [ListSizeBetween](../geneticengine/metahandlers/lists.py) metahandler.

To introduce a type-specific mutation or crossover, one should create a metahandler with a mutation or/and crossover method. The methods should take in the same parameters as the methods in [ListSizeBetween](../geneticengine/metahandlers/lists.py), and they should return a single individual of the same base type the metahandler defines.
