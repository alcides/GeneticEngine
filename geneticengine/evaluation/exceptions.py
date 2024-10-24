from geneticengine.solutions.individual import Individual


class IndividualFoundException(Exception):
    def __init__(self, individual: Individual):
        self.individual = individual
