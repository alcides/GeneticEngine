# Generation steps of the GP Algorithm

In this Folder we implement some Genetic Operators, these include crossover, mutation and selection methods.
Inside crossover.py file, we defined a method that apply the crossover operator between two individuals.
On the other hand, in mutation.py file,  we difined two diferent methods that apply the mutation operator, on for GP Algorithm and another for HillClimbing Algorithm.
And lastly we have the selection.py file, where we defined two parent selection algorithm (tournament selection and lexicase selection[[1]](#1)) along with an elitism function that is reponsible for returning the elite Individuals of a generation and a function that create novelty Individuals


## References

<a id="1">[1]</a>
T. Helmuth, L. Spector and J. Matheson, "Solving Uncompromising Problems With Lexicase Selection," in IEEE Transactions on Evolutionary Computation, vol. 19, no. 5, pp. 630-643, Oct. 2015.
