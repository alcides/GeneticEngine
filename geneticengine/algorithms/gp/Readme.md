Genetic Programming
==========

[Genetic Programming](https://d1wqtxts1xzle7.cloudfront.net/5871345/10.1.1.61.352-with-cover-page-v2.pdf?Expires=1666798486&Signature=anFva5JTM5cWB79aGPDqTEqFe93pd7sYaf5G8I4QT89V~Bbd3DViA9bhNvLVVIBL8TgWMEUKva~9FYgtHR50HIsnUqiPTexjK2fbCDlDayyVoGBGr9F7gUvTBa9AQJQ3tADMZ0sxwoIx-xto4ilqB4IocCVwoCzr1mNVBtvYODbNjZBSdTzBIZWWXDN16rqfKNzFjSu89heM7K2S-4dAAmyW9Vy5qi1QCybD3lal7~6z1Bv40wiPxUjgt9duIhStVlgInF6PXQFhvJxlVB6AZqp8AUlqBMJqJ0rB4HU2nkSjcnLgWljPRYg0sBcPVBr3dC-EHbZR8p8-AGKQcQuDRA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) was introduced by JR Koza in 1992. Our GP algorithm is grammar-guided, and therefore it is a GGGP algorithm. Still, it can be used as a standard GP method, using grammars to define the terminal and function class.

Our [GP](gp.py) class requires the following parameters:
* grammar (Grammar): The grammar used to guide the search.
* representation (Representation): The individual representation used by the GP program. The default is treebased_representation.
* problem (Problem): The problem we are solving. Either a SingleObjectiveProblem or a MultiObjectiveProblem.

Furthermore, the following optional parameters can be specified:
* favor_less_deep_trees (bool): If set to True, this gives a tiny penalty to deeper trees to favor simpler trees (default = False).
* randomSource (Callable[[int], RandomSource]): The random source function used by the program. Should take in an integer, representing the seed, and return a RandomSource. (default = random.Random)
* population_size (int): The population size (default = 200). Apart from the first generation, each generation the population is made up of the elites, novelties, and transformed individuals from the previous generation. Note that population_size > (n_elites + n_novelties + 1) must hold.
* n_elites (int): Number of elites, i.e. the number of best individuals that are preserved every generation (default = 5).
* n_novelties (int): Number of novelties, i.e. the number of newly generated individuals added to the population each generation. (default = 10).
* number_of_generations (int): Number of generations (default = 100).
* max_depth (int): The maximum depth a tree can have (default = 15).
* selection_method (Tuple[str, int]): Allows the user to define the method to choose individuals for the next population (default = ("tournament", 5)).

* probability_crossover (float): probability that an individual is chosen for cross-over [(default = 0.9)](https://link.springer.com/article/10.1007/s10710-008-9073-y).
* probability_mutation (float): probability that an individual is mutated [(default = 0.01)](https://link.springer.com/article/10.1007/s10710-008-9073-y).
* either_mut_or_cro (float | None): Switch evolution style to do either a mutation or a crossover. The given float defines the chance of a mutation. Otherwise a crossover is performed. (default = None),
* hill_climbing (bool): Allows the user to change the standard mutation operations to the hill-climbing mutation operation, in which an individual is mutated to 5 different new individuals, after which the best is chosen to survive (default = False).
* specific_type_mutation (type): Specify a type that is given preference when mutation occurs (default = None),
* specific_type_crossover (type): Specify a type that is given preference when crossover occurs (default = None),
* depth_aware_mut (bool): If chosen, mutations are depth-aware, giving preference to operate on nodes closer to the root. (default = True).
* depth_aware_co (bool): If chosen, crossovers are depth-aware, giving preference to operate on nodes closer to the root. (default = True).

* force_individual (Any): Allows the incorporation of an individual in the first population (default = None).
* seed (int): The seed of the RandomSource (default = 123).

* timer_stop_criteria (bool): If set to True, the algorithm is stopped after the time limit (default = 60 seconds). Then the fittest individual is returned (default = False).
* timer_limit (int): The time limit of the timer.

* save_to_csv (str): Saves a CSV file with the details of all the individuals of all generations.
* save_genotype_as_string (bool): Turn this off if you don't want to safe all the genotypes as strings. This saves memory and a bit of time.
* test_data (Any): Give test data (format: (X_test, y_test)) to test the individuals on test data during training and save that to the csv (default = None).
* only_record_best_inds (bool): Specify whether one or all individuals are saved to the csv files (default = True).

* callbacks (List[Callback]): The callbacks to define what is done with the returned prints from the algorithm (default = []).

A further discussion on the generational steps (mutation, crossover and selection) can be found [here](generation_steps/).
