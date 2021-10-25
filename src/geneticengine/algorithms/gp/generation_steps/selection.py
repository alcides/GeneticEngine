from typing import List, Callable
from geneticengine.core.random.sources import RandomSource
from geneticengine.algorithms.gp.Individual import Individual

def create_tournament(tournament_size: int, minimize=False) -> Callable[[RandomSource, List[Individual], int], List[Individual]]:
    
    def tournament(r: RandomSource, population: List[Individual], n: int) -> List[Individual]:
        winners = []
        for _ in range(n):
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