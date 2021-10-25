from typing import List, Callable
from copy import deepcopy
from geneticengine.core.random.sources import RandomSource
from geneticengine.algorithms.gp.Individual import Individual

def create_tournament(tournament_size: int, minimize=False) -> Callable[[RandomSource, List[Individual], int], List[Individual]]:
    
    def tournament(r: RandomSource, population: List[Individual], n_winners: int) -> List[Individual]:
        winners = []
        for _ in range(n_winners):
            candidates = [r.choice(population) for _ in range(tournament_size)]
            winner = candidates[0]
            for o in candidates[1:]:
                if o.fitness > winner.fitness and not minimize:
                    winner = o
                if o.fitness < winner.fitness and minimize:
                    winner = o
            winners.append(winner)
        return winners

    return tournament


def create_elitism(n_elites: int) -> Callable[[List[Individual]], List[Individual]]:
    
    def elitism(population: List[Individual]) ->  List[Individual]:
        population = population.sort(key=Individual.fitness)
        return [deepcopy(x) for x in population[: n_elites]]
    
    return elitism


def create_novelties(n_novelties: int, create_individual: Callable[[], Individual]) -> Callable[[], List[Individual]]:
    
    def novelties() ->  List[Individual]:
        return [create_individual() for _ in range(n_novelties)]
    
    return novelties
