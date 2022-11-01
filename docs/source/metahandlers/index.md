# Introduction to Metahandlers

##  Motivation

GGGP solution spaces often include base types that are not easily represented in BNF syntax. For example, integers are often represented as follows:
```
<I> ::= <P>
      | -<P>
<P> ::= <D><P>
<D> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 9
```
Not only is this cumbersome, this can result to issues(like integer overflowing.  In our STGP front-end approach, the user does not need to deal with this issue, as only the base type needs to be specified, and Genetic Engine handles the rest.

```python
class I:
	value: int
```

However, further restriction of the base types can be useful (maybe we only want positive integers/even numbers or floats between 0 and 1, in the case of probabilities). To allow the user this flexibility in Genetic Engine, we introduce metahandlers, inspired by [Refinement Types GP](https://link.springer.com/chapter/10.1007/978-3-030-58115-2_2). Metahandlers also allow the user to implement specific mutation and crossover methods for each base type in the grammar.

Here we only discuss the usage of metahandlers. For implementation details we refer to section 3.2 in [this paper](https://arxiv.org/abs/2210.04826).

## Usage

```
@dataclass
class Container(Parent):
	age : Annotated[int, IntRange[0, 200]]
	height : Annotated[float, FloatRange[0.10, 2.5]]
	children : Annotated[List[Container], ListSizeBetween[1,30]]
	var_name : Annotated[str, VarRange[["x", "y", "z"]]]
```

See [the full library of MetaHandlers](api.md) for more examples.
