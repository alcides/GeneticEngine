# Individual representations
Genetic Engine currently supports the following individual representations:
* tree-based representation, also known as Context-Free Grammars GP (CFG-GP)[[1]](#1)
* Grammatical Evolution (GE)[[2]](#2)
* Structured GE (SGE)[[3]](#3)
* and dynamic SGE (dSGE)[[4]](#4).

The representation can be chosen by the user. There are many discussions on which representation performs better as a search algorithm (fitness progression will differ across algorithms). Genetic Engine uses the same method for tree generation in CFG-GP and genotype-to-phenotype mapping in GE, SGE and dSGE, making it individual-representation independent on the implementation side. Still, we aim to implement performance enhancements on trees, benefitting the performance of CFG-GP, both on the time performance side (such as detailed in [[5]](#5)), as on the algorithm side.

We have implemented multiple tree-initialization methods: [Grow, Full, Ramped Half and Half, and Position Independent Grow](tree/treebased.py). The default we use is Ramped Half and Half.

## References

<a id="1">[1]</a>
Whigham, Peter A. "Grammatically-based genetic programming." Proceedings of the workshop on genetic programming: from theory to real-world applications. Vol. 16. No. 3. 1995.

<a id="2">[2]</a>
Ryan, Conor, John James Collins, and Michael O. Neill. "Grammatical evolution: Evolving programs for an arbitrary language." European conference on genetic programming. Springer, Berlin, Heidelberg, 1998.

<a id="3">[3]</a>
Lourenço, Nuno, Francisco B. Pereira, and Ernesto Costa. "SGE: a structured representation for grammatical evolution." International Conference on Artificial Evolution (Evolution Artificielle). Springer, Cham, 2015.

<a id="4">[4]</a>
Lourenço, Nuno, et al. "Structured grammatical evolution: a dynamic approach." Handbook of Grammatical Evolution. Springer, Cham, 2018. 137-161.

<a id="5">[5]</a>
Fonseca, Alcides. "Benchmarking Representations of Individuals in Grammar-Guided Genetic Programming." Evo* 2022: 5.
