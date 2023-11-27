from types import SimpleNamespace
from simanneal import Annealer



DesignSpace = SimpleNamespace


class AcceleratorAnnealer(Annealer):
    def __init__(self, initial_state=None, load_state=None):
        super().__init__(initial_state, load_state)


class AcceleratorOptimizer:
    def __init__(self, design_space) -> None:
        self.design_space = design_space
        self._optimizer = Annealer



