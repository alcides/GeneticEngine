from abc import ABC

    
class Statement(ABC):
    def evaluate(self, x: float) -> float:
        return x
    
class Expr(ABC):
    def evaluate(self, x: float) -> float:
        return 0.0
    
class Condition(ABC):
    def evaluate(self, x: bool) -> bool:
        return False

