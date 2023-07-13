# GeneticEngine

GeneticEngine is a framework for using Genetic Programming, empowered by Domain Knowledge, provided by the user.

In GeneticEngine, the user specifies the intended shape of their program using type annotations. Terminals and non-terminals can be defined using regular classes (although we recommend dataclasses), while abstract classes can be used to specify different options.

```python
class MyExpr(ABC):
	def eval(self):
		...

@dataclass
class Plus(MyExpr):
	left: MyExpr
	right: MyExpr

	def eval(self):
		return self.left.eval() + self.right.eval()

@dataclass
class Literal(MyExpr):
	value: int

	def eval(self):
		return self.value
```

In this example, we are defining the language that supports the plus operator and integer literals. GeneticEngine will be able to automatically generate all possible expressions, such as `Plus(left=Plus(left=Literal(12), right=Literal(12)), right=Literal(15))`. You can override `__str__` to have a more readable version of the tree.

From the definition of a grammar, and a fitness function like the one below, GeneticEngine can search for an expression that minimizes (or maximizes) the given function.

```python
def fitness_function(e:MyExpr) -> int
	return abs(42 - e.eval())
```

And then you can the program using:

```python
grammar = extract_grammar([Literal, Plus], MyExpr)
alg = SimpleGP(
    grammar,
    fitness_function,
    minimize=True,
    population_size=200,
    number_of_generations=200,
)
(b, bf) = alg.evolve()
print(bf, b)
```

To learn more about how to use Genetic Engine, you can follow [the tutorial](tutorial.md).

If you want to use GeneticEngine as a classifier or a regressor, you can use our [sklearn-compatible api](sklearn.md).


## Documentation


```{toctree}
---
maxdepth: 2
---
tutorial
sklearn
grammars
metahandlers/index
metahandlers/api
metahandlers/define
problems
algorithms
stopping
representations
genetic_operators
optimizations
autoapi/index
```



## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
