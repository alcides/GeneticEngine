# Tutorial


To use GeneticEngine to solve a Genetic Programming problem, you need two things:

* Define the tree structure of the representation
* Define a fitness function for each tree

Let us address each step in order.

Consider the example of Vectorial GP in [A Vectorial Approach to Genetic Programming](https://www.researchgate.net/publication/332309668_A_Vectorial_Approach_to_Genetic_Programming), in which values can be of type Element or type Vectorial.

We will use the following dataset, in which the first two columns are scalars, the following two are vectorial, and the last one is the target value. We will apply regression to the data set.

```python
dataset = [
    [1, 5, [1, 2, 3], [1, 5, 6], 0],
    [2, 2, [3, 2, 3], [1, 8, 6], 1],
    [1, 6, [2, 2, 3], [1, 5, 9], 2],
    [2, 8, [3, 2, 3], [1, 6, 6], 3],
]
```

At the root of our expression, we are looking for an element (because we are doing regression), but not all variables are elements — some are vectorial. To translate vectorial variables to elements, we can use some functions (mean, max, min, etc...). Some other functions convert vectorial into vectorial variables (such as cumulative sum).


## Step 0. Installation

You can install the dependencies from the requirements.txt file.

```
python -m pip install -r requirements.txt
```

The following snippets are taken from `examples/vectorialgp_example.py`, where the full running code, with imports, is available.

## Step 1. Object-Oriented Representation

Thus, we can create our class-based representation. We start by our generic types. These classes have no information, they are just used to represent the type of data.
They must inherit from Abstract Base Class (ABC), which make these classes as abstract and not instantiable, unlike their child classes.

```python
from abc import ABC

class Scalar(ABC):
    pass

class Vectorial(ABC):
    pass

```

Then, we create some terminals:


```python
@dataclass
class Value(Scalar):
    value: float


@dataclass
class ScalarVar(Scalar):
    index: Annotated[int, IntRange(0, 1)]


@dataclass
class VectorialVar(Vectorial):
    index: Annotated[int, IntRange(2, 3)]

```

The Value class corresponds to a literal scalar, like 3.4 or 3.2. ScalarVar corresponds to the features that are scalar values, which we know from the dataset that are columns at indices 0 and 1. Columns at indices 2 and 3 are vectorial, so we create another tree node that extends the Vectorial class.

We use the `dataclass` decorator to have automatic constructors, based on the properties of the class. The `Annotated` type allows to refine the type of base types (like `int`) with what we call MetaHandlers, which restrict the generation of random values. In this case we are using `IntRange` to restrict the possible values of index. We will talk more about metahandlers [here](metahandlers/index.md).

We now move to the non-terminals:

```python
@dataclass
class Add(Scalar):
    right: Scalar
    left: Scalar
```

The Add class represents the sum between two scalars. `right` and `left` can have any scalar expression, regardless of whether it is terminal or non-terminal (like another instance of Add).


```python
@dataclass
class Mean(Scalar):
    arr: Vectorial
```

Mean is an example of an operation that converts a vectorial expression into a scalar one.

```python
@dataclass
class CumulativeSum(Vectorial):
    arr: Vectorial
```

Similarly, CumulativeSum takes a vectorial expression (`arr`) and returns another vectorial expression.

The example we just wrote could have been written in a Grammar-Guided GP approach. In fact, we automate that translation:

```python
g = extract_grammar([Value, ScalarVar, VectorialVar, Mean, CumulativeSum], Scalar)
print("Grammar: {}.".format(repr(g)))
```

The output should be:

```
Grammar: Grammar<Starting=Scalar,Productions=[Scalar -> Value(value: float)|ScalarVar(index: [0...2])|Add(right: Scalar, left: Scalar)|Mean(arr: Vectorial);Vectorial -> VectorialVar(index: [2...4])|CumulativeSum(arr: Vectorial)]>.
```

which can be reformatted as:

```
Scalar -> Value(float)
Scalar -> ScalarVar([0...2])
Scalar -> Add(Scalar, Scalar)
Scalar -> Mean(Vectorial)
Vectorial -> VectorialVar([2...4])
Vectorial -> CumulativeSum(Vectorial)
```

which is a typical grammar in GGGP.

Now we just need to implement a fitness function to have a full, functional GP engine working. But for now, let us fake that fitness function:

```python
def fitness_function(n):
    return 0
```

This fitness function will give all individuals the same fitness (0), turning this problem into a random search. This is helpful to debug the representation, but will have to be replaced by a proper fitness function later.

Now we can run Genetic Engine, parameterized with this grammar and this fitness function:

```python
alg = GP(g, fitness_function, treebased_representation, minimize=True, population_size=10, number_of_generations=5)
(b, bf) = alg.evolve()
print(bf, b)
```

The output:

```
BEST at 0 / 5 is 0
BEST at 1 / 5 is 0
BEST at 2 / 5 is 0
BEST at 3 / 5 is 0
BEST at 4 / 5 is 0
0 Mean(arr=VectorialVar(index=2))
```

Non-surprisingly, you can see that the best fitness in each generation was also 0. The best individual after the 5 generations was the mean of the vectorial var in column 2. Feel free to change the initial seed to see other generated trees. Even if you inspect all possible trees, you will never find a tree that does the Mean of a scalar variable, because that would not be type-safe.

## Step 2. Writing Fitness Function

Writing a proper fitness functions is often a challenge in GP. Because we have a tabular dataset, we just have to apply the program to the input data, and return the prediction error:

```python
def compile(p, line):
    pass # Not implemented yet

def fitness_function(n: Scalar):
    regressor = lambda l: compile(n, l)
    y_pred = [regressor(line) for line in dataset]
    y = [line[-1] for line in dataset]
    r = mean_squared_error(y, y_pred)
    return r
```

Starting by the fitness_function, it takes a program that it will compile into a regressor. That regression will take a line of the dataset and predict the value (we are now predicting 0 for each instance). We will compute the mean squared error (from `sklearn.metrics`) between the prediction of each line and the real value in the last column.

If we run again, we will now get a constant fitness value of 3.5, which makes sense because all predictors return 0 for all instances. Let us not improve the compile function to create a predictor that works according to the tree.

Now let us write a proper `compile` function:

```python
def compile(p, line):
    if isinstance(p, Value):
        return p.value
    elif isinstance(p, ScalarVar) or isinstance(p, VectorialVar):
        return line[p.index]
    elif isinstance(p, Add):
        return compile(p.left, line) + compile(p.right, line)
    elif isinstance(p, Mean):
        return np.mean(compile(p.arr, line))
    elif isinstance(p, CumulativeSum):
        return np.cumsum(compile(p.arr, line))
    else:
        raise NotImplementedError(str(p))
```

This version pattern matches on the type of the node and recursively computes the value for the tree, always taking into consideration the dataset line being processed. Note that we are using numpy functions where needed.


Finally, run with a larger population for a longer time, you will see the best fitness decrease over time:

```python
alg = GP(
    g,
    treebased_representation,
    fitness_function,
    minimize=True,
    seed=0,
    population_size=200,
    number_of_generations=200,
)
(b, bf) = alg.evolve()
print(bf, b)
```
And the final lines of the output:
```
BEST at 199 / 200 is 0.69
0.6882625764219203
```

While our compile function was implemented as a stand-alone function, it could have been implemented in methods inside each element, following a more OOP style.

Another aspect that could be improved, and is left as an exercice to the reader is to implement an `__str__()` method in each class, to have a more readable output.

A final note is that the compile function does not have to be an interpreter, and it can use compilers in other languages (such as Python or Java). In the case of Python, we can translate the tree to Python and then (ab)use the `eval` function to convert a string into a runnable function.

```python
def translate(p):
    if isinstance(p, Value):
        return str(p.value)
    elif isinstance(p, ScalarVar) or isinstance(p, VectorialVar):
        return "line[{}]".format(p.index)
    elif isinstance(p, Add):
        return "({} + {})".format(translate(p.left), translate(p.right))
    elif isinstance(p, Mean):
        return "np.mean({})".format(translate(p.arr))
    elif isinstance(p, CumulativeSum):
        return "np.cumsum({})".format(translate(p.arr))
    else:
        raise NotImplementedError(str(p))


def fitness_function_alternative(n: Scalar):
    code = "lambda line: {}".format(translate(n))
    regressor = eval(code)
    y_pred = [regressor(line) for line in dataset]
    y = [line[-1] for line in dataset]
    r = mean_squared_error(y, y_pred)
    return r
```

In this example, the `translate` function converts the tree into Python code, and in the beginning of the `fitness_function_alternative` function, we use eval to convert that code string into a callable function.
