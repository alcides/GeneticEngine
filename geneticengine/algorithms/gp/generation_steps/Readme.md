# Generation steps of the GP Algorithm

In this Folder we implement some Genetic Operators, these include crossover, mutation and selection methods.
Inside crossover.py file, we defined a method that apply the crossover operator between two individuals.
On the other hand, in mutation.py file, we defined two different methods that apply the mutation operator, one for the GP algorithm and another for the hill climbing algorithm (which can also be used within GP).
And lastly we have the selection.py file, where we defined two parent selection algorithms (tournament selection and lexicase selection[[1]](#1)). Furthermore, we include an elitism function, responsible for returning the elite individuals of a generation, and a novelty function, responsible for creating novel individuals.


## References

<a id="1">[1]</a>
T. Helmuth, L. Spector and J. Matheson, "Solving Uncompromising Problems With Lexicase Selection," in IEEE Transactions on Evolutionary Computation, vol. 19, no. 5, pp. 630-643, Oct. 2015.
