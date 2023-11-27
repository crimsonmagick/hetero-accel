import logging
import random
from collections import namedtuple
from simanneal import Annealer


__all__ = ['AcceleratorState', 'DesignSpace', 'AcceleratorOptimizer']

logger = logging.getLogger(__name__)


AcceleratorState = namedtuple("AcceleratorState", 
                              ['pe_array_x', 'pe_array_y',
                               'precision', 'sram_size',
                               'ifmap_spad_size', 'weights_spad_size', 'psum_spad_size'])


class DesignSpace(AcceleratorState):
    """Wrapper for the design space of possible accelerator architectures
    """
    def __new__(cls, **kwargs):
        for key, value in kwargs.items():
            assert key in super()._fields
            assert isinstance(value, (list, tuple)) and len(value) > 0
        self = super(DesignSpace, cls).__new__(cls, **kwargs)
        return self

    def sample(self) -> AcceleratorState:
        """Get a random sample from the design space
        """
        values = {
            field: random.choice(getattr(self, field))
            for field in self._fields
        }
        #return super().__class__(**values)
        return AcceleratorState(**values)

    def extract(self, *args, **kwargs) -> AcceleratorState:
        """Extract a specific solution from the design space
        """
        assert len(args) == 0 or len(kwargs) == 0, "Only one type of input is supported"
        if len(args) > 0:
            values_to_get = args[0] if isinstance(args[0], list) and len(args) == 1 else args
        elif len(kwargs) > 0:
            values_to_get = [kwargs[field] for field in self._fields]
        else:
            raise ValueError("No inputs were given")
        
        for value, field in zip(values_to_get, self._fields):
            assert value in getattr(self, field), f"Invalid value {value} for {field}"

        return AcceleratorState(*values_to_get)


class AcceleratorOptimizer(Annealer):
    """Wrapper for Simulated Annealing optimizer
    """
    def __init__(self, simanneal_args, accelerator, design_space, workload, accuracy_lut) -> None:
        self.accelerator = accelerator
        self.design_space = design_space
        self.workload = workload
        self.accuracy_lut = accuracy_lut

        inital_state = self.design_space.extract(self.accelerator.width,
                                                 self.accelerator.height,
                                                 self.accelerator.precision_weights,
                                                 self.accelerator.glb_sram_size,
                                                 self.accelerator.ifmap_spad_size,
                                                 self.accelerator.weights_spad_size,
                                                 self.accelerator.psum_spad_size)
        super().__init__(initial_state, getattr(args, 'simanneal_load_state', None))

        self.copy_strategy = 'deepcopy'
        if args is None or \
            any(getattr(arg, args, None) is None
                for arg in ['simanneal_Tmax', 'simanneal_Tmin', 'simanneal_steps', 'simanneal_updates']):

            self.set_schedule(self.auto(minutes=10))

        else:
            self.Tmax = args.simanneal_Tmax
            self.Tmin = args.simaneal_Tmin
            self.steps = args.simanneal_steps
            self.updates = args.simanneal_updates

        # TODO: Include simanneal args
        # TODO: Include timeloop wrapper object, look at previous exploration (compression)
        # TODO: Fill move and energy methods

    def run(self):
        """Run Simulated Annealing
        """
        best_state, best_fitness = super().anneal()
        self.best_solution = best_state
        self.best_solution_fitness = best_fitness

    @property
    def best_solution(self):
        return self.best_solution

    @property
    def best_solution_fitness(self):
        return self.best_solution_fitness

    def update(self):
        """Log the results of the exploration (override from parent)
        """
        raise NotImplementedError

    def move(self):
        """Alter the current state
        """
        raise NotImplementedError

    def energy(self):
        """Evaluate the fitness of the current state
        """
        raise NotImplementedError


if __name__ == "__main__":
    a = DesignSpace(pe_array_x=list(range(10)),
                    pe_array_y=list(range(10)),
                    precision=list(range(10)),
                    sram_size=list(range(10)),
                    ifmap_spad_size=list(range(10)),
                    weights_spad_size=list(range(10)),
                    psum_spad_size=list(range(10)))

    print(a)
    print(a.sample())
    print(a.extract([1, 2, 3, 4, 5, 6, 7]))

