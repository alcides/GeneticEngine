# Individual representations
Genetic Engine currently supports 4 individual representations:

* Tree-based representation, also known as Context-Free Grammars GP (CFG-GP)[^1]
* Grammatical Evolution (GE)[^2]
* Structured GE (SGE)[^3]
* Dynamic SGE (dSGE)[^4]

The representation can be chosen by the user. There are many discussions on which representation performs better as a search algorithm (fitness progression will differ across algorithms). Genetic Engine uses the same method for tree generation in CFG-GP and genotype-to-phenotype mapping in GE, SGE and dSGE, making it individual-representation independent on the implementation side. Still, we aim to implement performance enhancements on trees, benefitting the performance of CFG-GP, both on the time performance side (such as detailed in [^5]), as on the algorithm side.

## Tree-based

```{eval-rst}
.. autoapiclass:: geneticengine.core.representations.tree.treebased.TreeBasedRepresentation
```

## Grammatical Evolution

```{eval-rst}
.. autoapiclass:: geneticengine.core.representations.grammatical_evolution.ge.GrammaticalEvolutionRepresentation
```

## Structured Grammatical Evolution

```{eval-rst}
.. autoapiclass:: geneticengine.core.representations.grammatical_evolution.structured_ge.StructuredGrammaticalEvolutionRepresentation
```

## Dynamic Structured Grammatical Evolution

```{eval-rst}
.. autoapiclass:: geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge.DynamicStructuredGrammaticalEvolutionRepresentation
```

## Stack-based Grammatical Evolution

```{eval-rst}
.. autoapiclass:: geneticengine.core.representations.stackgggp.StackBasedGGGPRepresentation
```

## Probabilistic Grammatical Evolution (PGE)

Genetic Engine supports PGE. By some, PGE is recognized as an individual-representation method, even though the PGE concept can be applied to any of the above-mentioned individual-representation methods. As such, we have included an explanation in the [grammars section](grammars.md).

##### References

[^1]: Whigham, Peter A. "Grammatically-based genetic programming." Proceedings of the workshop on genetic programming: from theory to real-world applications. Vol. 16. No. 3. 1995.

[^2]: Ryan, Conor, John James Collins, and Michael O. Neill. "Grammatical evolution: Evolving programs for an arbitrary language." European conference on genetic programming. Springer, Berlin, Heidelberg, 1998.

[^3]: Lourenço, Nuno, Francisco B. Pereira, and Ernesto Costa. "SGE: a structured representation for grammatical evolution." International Conference on Artificial Evolution (Evolution Artificielle). Springer, Cham, 2015.

[^4]: Lourenço, Nuno, et al. "Structured grammatical evolution: a dynamic approach." Handbook of Grammatical Evolution. Springer, Cham, 2018. 137-161.

[^5]: Ingelse, Leon, et al. "Benchmarking Representations of Individuals in Grammar-Guided Genetic Programming." Evo* 2022: 5.
