Genetic Engine
==============

[![Documentation](https://readthedocs.org/projects/geneticengine/badge/?version=latest)](https://geneticengine.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/alcides/GeneticEngine/branch/main/graph/badge.svg?token=ZX84SA7IEP)](https://codecov.io/gh/alcides/GeneticEngine)

A hybrid between strongly-typed (STGP) and grammar-guided genetic programming (GGGP).

About GeneticEngine
-------------------

GeneticEngine is a Genetic Programming framework for single- and multi-objective optimization. GeneticEngine allows the user to provide domain knowledge about the shape of the solution (using type annotations) and by defining the fitness function.


Documentation
-------------

* [Documentation](https://geneticengine.readthedocs.io/)
* [Examples](examples/)

```python
class MyExpr(ABC):
	"MyExpr is a non-terminal/abstract class."
	def eval(self):
		...

@dataclass
class Plus(MyExpr):
	"E -> E + E"
	left: MyExpr
	right: MyExpr

	def eval(self):
		return self.left.eval() + self.right.eval()

@dataclass
class Literal(MyExpr):
	"E -> <int>"
	value: int

	def eval(self):
		return self.value
```

In this small example, we are defining the language that supports the plus operator and integer literals. GeneticEngine will be able to automatically generate all possible expressions, such as `Plus(left=Plus(left=Literal(12), right=Literal(12)), right=Literal(15))`, and guide the search towards your goal (e.g., `lambda x: abs(x-2022)`). For this very simple toy problem, it will find an expression that computes 2022, ideally as small as possible. And this is a very uninteresting example. But if you introduce variables into the mix, you have a very powerful symbolic regression toolkit for arbitrarily complex expressions.


Contributing
-------------

After cloning the repo, please run `source setup_dev.sh` to install virtualenv, all dependencies and setup all pre-commit hooks.

Pull Requests are more than welcome!


Authors
----------
GeneticEngine has been developed at [LASIGE](https://www.lasige.pt), [University of Lisbon](https://ciencias.ulisboa.pt) by:

* [Alcides Fonseca](http://alcidesfonseca.com)
* [Leon Ingelse](https://leoningel.github.io)
* [Guilherme Espada](https://www.lasige.di.fc.ul.pt/user/732)
* [Paulo Santos](https://pcanelas.com/)
* [Pedro Barbosa](https://www.lasige.di.fc.ul.pt/user/661)
* [Eduardo Madeira](https://www.lasige.pt/member/jose-eduardo-madeira)

Acknowledgements
----------------

This work was supported by Fundação para a Ciência e Tecnologia (FCT) through:

* [the LASIGE Research Unit](https://www.lasige.pt) (ref. UIDB/00408/2020 and UIDP/00408/2020)
* Pedro Barbosa PhD fellowship (SFRH/BD/137062/2018)
* Guilherme Espada PhD fellowship (UI/BD/151179/2021)
* Paulo Santos CMU|Portugal PhD fellowship (SFRH/BD/151469/2021)
* [the FCT Exploratory project RAP](http://wiki.alcidesfonseca.com/research/projects/rap/) (EXPL/CCI-COM/1306/2021)
* the FCT Advanced Computing projects (2022.15800.CPCA.A1, CPCA/A1/395424/2021, CPCA/A1/5613/2020, CPCA/A2/6009/2020)

And by Lisboa2020, Compete2020 and FEDER through:

* [the CMU|Portugal CAMELOT project](http://wiki.alcidesfonseca.com/research/projects/camelot/) (LISBOA-01-0247-FEDER-045915)


Publications
-----------------

* [Comparing the expressive power of Strongly-Typed and Grammar-Guided Genetic Programming](https://www.researchgate.net/publication/370277603_Comparing_the_expressive_power_of_Strongly-Typed_and_Grammar-Guided_Genetic_Programming) at GECCO'23
* [Data types as a more ergonomic frontend for Grammar-Guided Genetic Programming](https://arxiv.org/pdf/2210.04826) at GPCE'22
* [Grammatical Evolution Mapping for Semantically-Constrained Genetic Programming](https://www.researchgate.net/profile/Alcides-Fonseca/publication/358528379_Grammatical_Evolution_Mapping_for_Semantically-Constrained_Genetic_Programming/links/620a1ecf634ff774f4cc2cee/Grammatical-Evolution-Mapping-for-Semantically-Constrained-Genetic-Programming.pdf) at GPTP'21
* [The Usability Argument for Refinement Typed Genetic Programming](https://link.springer.com/chapter/10.1007/978-3-030-58115-2_2) at PPSN'20

Applications of GeneticEngine
-----------------------------

* [Comparing Individual Representations in Grammar-Guided Genetic Programming for Glucose Prediction in People with Diabetes](https://www.researchgate.net/publication/371324298_Comparing_Individual_Representations_in_Gram-mar-Guided_Genetic_Programming_for_Glucose_Prediction_in_People_with_Diabetes) at Grammatical Workshop at GECCO'23
* [Domain-Aware Feature Learning with Grammar-Guided Genetic Programming](https://link.springer.com/chapter/10.1007/978-3-031-29573-7_15) at EuroGP'23
* [Benchmarking Individual Representation in Grammar-Guided Genetic Programming](https://wwwww.easychair.org/publications/preprint_download/wqrb) at Evo*'22


Let us know if your paper uses Genetic Engine, to list it here.

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
  booktitle = {{GPCE} '22: Concepts and Experiences, Auckland, NZ, December 6 - 7, 2022},
  pages     = {1},
  publisher = {{ACM}},
  year      = {2022},
}
```
