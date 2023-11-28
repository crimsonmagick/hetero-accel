import logging
import random
import os.path
from copy import deepcopy
from collections import namedtuple
from simanneal import Annealer
from src.scheduler import MultiDNNScheduler
from src.timeloop import TimeloopWrapper


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
    WORST_ENERGY = -1.0

    def __init__(self,
                 simanneal_args,
                 num_accelerators,
                 accelerator_cfg,
                 design_space,
                 workload,
                 accuracy_lut,
                 hw_constraints
        ):
        self.num_accelerators = num_accelerators
        self.accelerator_cfg = accelerator_cfg
        self.design_space = design_space
        self.workload = workload
        self.accuracy_lut = accuracy_lut
        self.hw_constraints = hw_constraints
        self.energy_dict = {}
        self.latency_dict = {}
        self.area_dict = {}
        self.logdir = simanneal_args.logdir

        # initialize timeloop
        self.init_timeloop(simanneal_args.layer_type_whitelist)
        # initialize scheduler
        self.init_scheduler()

        # use the default accelerator as the initial state of the exploration
        initial_state = self.design_space.extract(self.accelerator_cfg.width,
                                                  self.accelerator_cfg.height,
                                                  self.accelerator_cfg.precision_weights,
                                                  self.accelerator_cfg.glb_sram_size,
                                                  self.accelerator_cfg.ifmap_spad_size,
                                                  self.accelerator_cfg.weights_spad_size,
                                                  self.accelerator_cfg.psum_spad_size)
        initial_state = [initial_state] * self.num_accelerators
        super().__init__(initial_state, getattr(simanneal_args, 'simanneal_load_state', None))

        # get baseline measurements
        assert self.energy() != self.WORST_ENERGY, "Baseline produces negative energy"
        self.initial_energy = self.latest_energy
        self.initial_latency = self.latest_latency
        self.initial_area = self.latest_area
        logger.debug(f"SA: Initial state={self.state} -> Energy={self.initial_energy}, "
                     f"Latency={self.initial_latency}, Area={self.initial_area}")

        # setup scheduling parameters during annealing
        self.update = self._update_logging
        self.copy_strategy = 'deepcopy'
        if simanneal_args.simanneal_auto_schedule or \
            simannea_args is None or \
            any(getattr(arg, simanneal_args, None) is None
                for arg in ['simanneal_Tmax', 'simanneal_Tmin', 'simanneal_steps', 'simanneal_updates']):
            # automatic annealing schedule
            self.set_schedule(self.auto(minutes=10))
        else:
            # user-defined annealing schedule
            self.Tmax = simanneal_args.simanneal_Tmax
            self.Tmin = simanneal_args.simaneal_Tmin
            self.steps = simanneal_args.simanneal_steps
            self.updates = simanneal_args.simanneal_updates

    def init_timeloop(self, layer_type_whitelist):
        """Initialize timeloop wrapper object
        """
        tl_workdir = os.path.join(self.logdir, 'timeloop_simanneal')
        self.timeloop_wrapper = TimeloopWrapper(self.accelerator_cfg.type, tl_workdir)

        # prepare each layer for timeloop simulations
        layer_idx = 0
        self.timeloop_problems_per_dnn = {}
        for arch, net_wrapper in self.workload.dnns.items():
            self.timeloop_problems_per_dnn[arch] = []

            layers_to_consider = [name for name, module in net_wrapper.model.named_modules()
                                  if isinstance(module, layer_type_whitelist)]
            for layer_name, layer_info in self.workload.get_summary(arch).items():
                if layer_name not in layers_to_consider:
                    continue

                problem_name = f'{arch}__layer{layer_idx}_{layer_name}'
                self.timeloop_problems_per_dnn[arch].append(problem_name)
                problem_filepath = os.path.join(self.timeloop_wrapper.workload_dir, problem_name + '.yaml')
                self.timeloop_wrapper.init_problem(problem_name,
                                                   layer_info.layer_type,
                                                   layer_info.dimensions,
                                                   problem_filepath)
                layer_idx += 1

    def init_scheduler(self):
        """Initialize the design-time model of the scheduler
        """
        self.scheduler = MultiDNNScheduler(self.hw_constraints.deadline)

    def run(self):
        """Run Simulated Annealing
        """
        self.anneal()
        self.save_state(os.path.join(self.logdir, f'simanneal_energy_{self.best_energy}.state'))

    def _update_logging(self, step, T, E, acceptance, improvement):
        """Log the results of the exploration (override from parent)
        """
        def time_string(seconds):
            """Returns time in seconds as a string formatted HHHH:MM:SS."""
            s = int(round(seconds))  # round to nearest second
            h, s = divmod(s, 3600)   # get hours and remainder
            m, s = divmod(s, 60)     # split remainder into minutes and seconds
            return '%4i:%02i:%02i' % (h, m, s)

        elapsed = time.time() - self.start
        if step == 0:
            logger.info('\n Temperature        Energy    Accept   Improve     Elapsed   Remaining')
            logger.info('\r{Temp:12.5f}  {Energy:12.2f}                      {Elapsed:s}            '
                        .format(Temp=T, Energy=E, Elapsed=time_string(elapsed)))
        else:
            remain = (self.steps - step) * (elapsed / step)
            logger.info('\r{Temp:12.5f}  {Energy:12.2f}   {Accept:7.2%}   {Improve:7.2%}  '
                        '{Elapsed:s}  {Remaining:s}'
                        .format(Temp=T, Energy=E, Accept=acceptance, Improve=improvement,
                                Elapsed=time_string(elapsed), Remaining=time_string(remain)))

    def _update_stderr(self, step, T, E, acceptance, improvement):
        """Direct copy from the update function of Annealer
           https://github.com/perrygeo/simanneal/blob/master/simanneal/anneal.py#L127
        """
        def time_string(seconds):
            """Returns time in seconds as a string formatted HHHH:MM:SS."""
            s = int(round(seconds))  # round to nearest second
            h, s = divmod(s, 3600)   # get hours and remainder
            m, s = divmod(s, 60)     # split remainder into minutes and seconds
            return '%4i:%02i:%02i' % (h, m, s)

        elapsed = time.time() - self.start
        if step == 0:
            print('\n Temperature        Energy    Accept   Improve     Elapsed   Remaining',
                  file=sys.stderr)
            print('\r{Temp:12.5f}  {Energy:12.2f}                      {Elapsed:s}            '
                  .format(Temp=T,
                          Energy=E,
                          Elapsed=time_string(elapsed)),
                  file=sys.stderr, end="")
            sys.stderr.flush()
        else:
            remain = (self.steps - step) * (elapsed / step)
            print('\r{Temp:12.5f}  {Energy:12.2f}   {Accept:7.2%}   {Improve:7.2%}  {Elapsed:s}  {Remaining:s}'
                  .format(Temp=T,
                          Energy=E,
                          Accept=acceptance,
                          Improve=improvement,
                          Elapsed=time_string(elapsed),
                          Remaining=time_string(remain)),
                  file=sys.stderr, end="")
            sys.stderr.flush()

    def move(self):
        """Alter the current state, by changing at least one accelerator
        """
        new_state = [self.design_space.sample() for _ in range(self.num_accelerators)]
        while new_state == self.state:
            new_state = [self.design_space.sample() for _ in range(self.num_accelerators)]
        self.state = new_state

    def energy(self):
        """Evaluate the fitness of the current state
        """
        def area_constraint_satisfied():
            raise NotImplementedError
            return area

        def accuracy_constraint_satisfied(arch, precision):
            return self.accuracy_lut.loc[
                        (self.accuracy_lut['Network'] == arch) &&
                        (self.accuracy_lut['QuantBits'] == precision)
                    ]['Valid'].iloc[0] == 1

        # metrics to be accumulated
        energy_dict = {}
        area_dict = {}
        latency_dict = {}

        # iterate over each accelerator
        for accelerator in self.state:
            # iterate over each DNN
            for arch in self.workload.dnns.keys():
                # check accuracy constraint
                if not accuracy_constraint_satisfied(arch, accelerator.precision):
                    continue
                # check if this evaluation was executed before
                if (arch, accelerator) in self.energy_dict:
                    continue

                # adjust timeloop with the accelerator parameters
                raise NotImplementedError

                energy_dict[(arch, accelerator)] = 0
                latency_dict[(arch, accelerator)] = 0
                area_dict[(arch, accelerator)] = 0
                # iterate over each timeloop problem (layer) of the DNN
                for problem_name in self.timeloop_problems_per_dnn[arch]:

                    # run timeloop and get results
                    self.timeloop_wrapper.run(problem_name)
                    results = self.timeloop_wrapper.get_results()
                    energy_dict[(arch, accelerator)] += results.energy
                    latency_dict[(arch, accelerator)] += results.cycles
                    area_dict[(arch, accelerator)] += results.area

        # update stored metrics with executed evaluations
        self.energy_dict.update(energy_dict)
        self.latency_dict.update(area_dict)
        self.area_dict.update(area_dict)

        # perform the scheduling and get a concrete DNN-to-accelerator mapping
        mapping, deadline_satisfied = self.sceduler.run(dnns=list(self.workload.dnns.keys()),
                                                        accelerators=self.state,
                                                        energy_dict=self.energy_dict,
                                                        latency_dict=self.latency_dict)

        # TODO: Figure out how to get results for energy, latency and area based on the mapping
        self.latest_energy = ''
        self.latest_latency = ''
        self.latest_area = ''

        if not deadline_satisfied or not area_constraint_satisfied(area):
            return self.WORST_ENERGY
        return self.latest_energy 


if __name__ == "__main__":
    a = DesignSpace(pe_array_x=list(range(10)),
                    pe_array_y=list(range(10)),
                    precision=list(range(10)),
                    sram_size=list(range(10)),
                    ifmap_spad_size=list(range(10)),
                    weights_spad_size=list(range(10)),
                    psum_spad_size=list(range(10)))
    b = DesignSpace(pe_array_x=list(range(10)),
                    pe_array_y=list(range(10)),
                    precision=list(range(10)),
                    sram_size=list(range(10)),
                    ifmap_spad_size=list(range(10)),
                    weights_spad_size=list(range(10)),
                    psum_spad_size=list(range(10)))


    #print(a == b)
    #print(a.sample())
    #print(a.extract([1, 2, 3, 4, 5, 6, 7]))

    prev_state = a.sample()
    new_state = a.sample()
    while new_state == prev_state:
        new_state = a.sample()

    print(prev_state)
    print(new_state)

