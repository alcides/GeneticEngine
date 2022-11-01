# Genetic Engine

A hybrid between strongly-typed (STGP) and grammar-guided genetic programming (GGGP).

## About Genetic Engine

Genetic Engine is a framework for using Genetic Programming in different contexts. Genetic Engine allows the user to define trees in terms of Classes and Inheritance, as they would in a regular programming environment. Our framework takes care of generating individuals, mutating them and crossing them over. The user also defines a fitness function that takes a tree and returns a fitness score. This often requires writing (or reusing) a tree interpreter, as it is custom in these types of approaches. We intend to include all GP specific parameters to Genetic Engine ([see all we have implemented](algorithms/genetic_programming.md)). If you don't see what you need, please create an issue, and we will add it as soon as possible.

Genetic Engine also supports [off-the-shelf sklearn-style classifiers and regressors](sklearn.md).

The main different between STGP and GGGP is that the restrictions on what trees are valid are done via types, while in GGGP they are expressed using a grammar. Genetic Engine extracts the grammar from the types and their relationship, allowing to use any technique from GGGP (such as Grammatical Evolution) in STGP.

The advantages of using STGP are:

* The user does not need to know grammars, EBNF or any other grammar syntax
* There is no need for a textual representation of programs, as trees can be the only representation (à lá lisp).
* There is no need for parsing a textual program to a tree, to then interpret the tree (unlike [PonyGE2](https://github.com/PonyGE/PonyGE2), which works on a textual level)
* Mutations and Recombination are automatically type-safe, where in a grammar that type-safety is implicit in the structure of the grammar (and hard to reason)


## Authors

GeneticEngine has been developed at [LASIGE](https://www.lasige.pt), [University of Lisbon](https://ciencias.ulisboa.pt) by:

* [Alcides Fonseca](http://alcidesfonseca.com)
* [Leon Ingelse](https://leoningel.github.io)
* [Guilherme Espada](https://www.lasige.di.fc.ul.pt/user/732)
* [Paulo Santos](https://pcanelas.com/)
* [Pedro Barbosa](https://www.lasige.di.fc.ul.pt/user/661)
* [Eduardo Madeira](https://www.lasige.pt/member/jose-eduardo-madeira)


Below you'll find a step-by-step guide on how to use Genetic Engine, together with an example. For more specific documentation on the implementation and algorithms available, follow the links below. If you cannot find the information you are looking for, please create an issue, and we will update as soon as possible.

* [Individual representation](representations.md)
* [Grammar specifics](grammars.md)
* [Implemented generations steps (mutation, crossover and selection)](genetic_operators.md)
* [Metahandlers](metahandlers.md)
* [Available algorithms](algorithms/)
* [Sklearn-style classifiers and regressors](sklearn.md)

## Acknowledgements

This work was supported by Fundação para a Ciência e Tecnologia (FCT) through:

* the LASIGE Research Unit (ref. UIDB/00408/2020 and UIDP/00408/2020)
* Pedro Barbosa PhD fellowship (SFRH/BD/137062/2018)
* Guilherme Espada PhD fellowship (UI/BD/151179/2021)
* Paulo Santos CMU|Portugal PhD fellowship (SFRH/BD/151469/2021)
* the CMU|Portugal CAMELOT project (LISBOA-01-0247-FEDER-045915)
* the FCT Exploratory project RAP (EXPL/CCI-COM/1306/2021)

Please cite as:
```
Espada, Guilherme, et al. "Data types as a more ergonomic frontend for Grammar-Guided Genetic Programming.", GPCE '22: Concepts and Experiences, 2022
```

Bibtex:
```
@inproceedings{espada2022data,
  author={Guilherme Espada and Leon Ingelse and Paulo Canelas and Pedro Barbosa and Alcides Fonseca},
  editor    = {Bernhard Scholz and Yukiyoshi Kameyama},
  title={Datatypes as a More Ergonomic Frontend for Grammar-Guided Genetic Programming},
  booktitle = {{GPCE} '22: Concepts and Experiences, Auckland, NZ, December 6
               - 7, 2022},
  pages     = {1},
  publisher = {{ACM}},
  year      = {2022},
}
```
