import logging
import random
import os.path
import sys
import pickle
from types import SimpleNamespace
from collections import OrderedDict
from time import time
from shutil import copy
from simanneal import Annealer
from src.scheduler import Scheduler
from src.timeloop import TimeloopWrapper
from src.utils import get_contents_table


__all__ = ['DesignSpace', 'AcceleratorOptimizer']

logger = logging.getLogger(__name__)


class DesignSpace(SimpleNamespace):
    """Wrapper for the design space of possible accelerator architectures
    """
    def __init__(self, accelerator_state_class, **kwargs):
        super().__init__(**kwargs)
        self._fields = list(self.__dict__.keys())
        self.accelerator_state_class = accelerator_state_class
        for key, value in kwargs.items():
            assert key in accelerator_state_class._fields, f'{key}'
            assert isinstance(value, (list, tuple)) and len(value) > 0

    # def __setattr__(self, key, val):
    #     """Emulating the functionality of a namedtuple"""
    #     raise AttributeError('Cannot set new values for DesignSpace')

    def sample(self, override_dict=None):
        """Get a random sample from the design space. A semi-random sample
           can be obtained be setting specific values to the override dict
        """
        override_dict = {} if override_dict is None else override_dict
        values = {
            field: random.choice(getattr(self, field))
                if field not in override_dict else override_dict[field]
            for field in self._fields
        }
        #return super().__class__(**values)
        return self.accelerator_state_class(**values)

    def extract(self, *args, **kwargs):
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

        return self.accelerator_state_class(*values_to_get)


class AcceleratorOptimizer(Annealer):
    """Wrapper for Simulated Annealing optimizer
    """
    WORST_ENERGY = -1.0

    def __init__(self,
                 args,
                 num_accelerators,
                 accelerator_cfg,
                 workload,
                 accuracy_lut,
                 hw_constraints
        ):
        self.num_accelerators = num_accelerators
        self.accelerator_cfg = accelerator_cfg
        self.workload = workload
        self.accuracy_lut = accuracy_lut
        self.hw_constraints = hw_constraints
        self.energy_dict = OrderedDict()
        self.latency_dict = OrderedDict()
        self.area_dict = OrderedDict()
        self.logdir = args.logdir
        self.design_space = DesignSpace(accelerator_cfg.state,
                                        **accelerator_cfg.design_space)

        # initialize timeloop
        self.init_timeloop(args.layer_type_whitelist)
        # initialize scheduler
        self.scheduler = Scheduler(args.scheduler_type)
        self.latest_schedule = None

        initial_state = self.get_initial_state(args)
        super().__init__(initial_state, getattr(args, 'simanneal_load_state', None))

        # get baseline measurements
        assert self.energy(initial=True) <= self.WORST_ENERGY, "Baseline produces negative energy"
        self.initial_energy = self.latest_energy
        self.initial_latency = self.latest_latency
        self.initial_area = self.latest_area
        logger.info(f"Initial results -> Energy={self.initial_energy:.3e}, "
                     f"Latency={self.initial_latency:.3e}, Area={self.initial_area:.3e}")

        # setup scheduling parameters during annealing
        self.update = self._update_logging
        self.copy_strategy = 'deepcopy'
        if args is None or \
            getattr(args, 'simanneal_auto_schedule', False) or \
            any(getattr(args, arg, None) is None
                for arg in ['simanneal_Tmax', 'simanneal_Tmin', 'simanneal_steps', 'simanneal_updates']):
            # automatic annealing schedule
            self.set_schedule(self.auto(minutes=10))
        else:
            # user-defined annealing schedule
            self.Tmax = args.simanneal_Tmax
            self.Tmin = args.simanneal_Tmin
            self.steps = args.simanneal_steps
            self.updates = args.simanneal_updates

    def get_initial_state(self, args):
        """Configure the initial state of the optimizer, w.r.t. the
           selected heterogeneity of he accelerator
        """
        initial_state = []

        if getattr(args, 'load_state_from', None) is not None and \
           os.path.exists(args.load_state_from):
            # load the initial state of the accelerator
            initial_state = self.load_state(args.load_state_from)

        else:
            # build a heterogeneous accelerator, with specific precision for each accelerator
            for accelerator_idx in range(self.num_accelerators):
                precision = self.accelerator_cfg.design_space['precision'][accelerator_idx]
                values = [
                    precision if 'precision' in field.lower() else getattr(self.accelerator_cfg, field)
                    for field in self.design_space.fields
                ]
                initial_state.append(self.design_space.extract(*values))

        logger.info("=> Initial state:")
        for state in initial_state:
            logger.info(f"\t{state}")
        return initial_state

    def init_timeloop(self, layer_type_whitelist):
        """Initialize timeloop wrapper object
        """
        tl_workdir = os.path.join(self.logdir, 'timeloop_simanneal')
        self.timeloop_wrapper = TimeloopWrapper(self.accelerator_cfg.type, tl_workdir)

        # prepare each layer for timeloop simulations
        self.timeloop_problems_per_dnn = {}
        for arch, net_wrapper in self.workload.dnns.items():
            self.timeloop_problems_per_dnn[arch] = []

            layers_to_consider = [name for name, module in net_wrapper.model.named_modules()
                                  if isinstance(module, layer_type_whitelist)]
            layer_idx = 0
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

        elapsed = time() - self.start
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
        # save schedule
        if self.latest_schedule is not None:
            self.latest_schedule.visualize()

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

        elapsed = time() - self.start
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

    def save_state(self, save_state_to=None):
        """Save the latest state and results from energy/fitness calculation
        """
        state_dict = {'energy': self.energy_dict,
                      'latency': self.latency_dict,
                      'area': self.area_dict,
                      'schedule': self.latest_schedule,
                      'state': self.state,
                      'constraints': getattr(self, 'hw_constraints', None),
                      'latest_energy': self.latest_energy,
                      'latest_latency': self.latest_latency,
                      'latest_area': self.latest_area}

        save_state_to = save_state_to or os.path.join(self.logdir, 'state.sa.pkl')
        with open(save_state_to, 'wb') as f:
            pickle.dump(state_dict, f)
        logger.info(f"Saved state in: {save_state_to}")

    def load_state(self, load_from, save_state_to=None):
        """Load the state and its results from a given file
        """
        with open(load_from, 'rb') as f:
            state_dict = pickle.load(f)
        logger.info(f"Loaded initial state from checkpoint ({load_from})")
        logger.info(f"Checkpoint contents:\n{get_contents_table(state_dict)}\n")
        save_state_to = save_state_to or os.path.join(self.logdir, 'state.sa.pkl')
        copy(load_from, save_state_to)

        self.energy_dict = state_dict['energy']
        self.latency_dict = state_dict['latency']
        self.area_dict = state_dict['area']
        if getattr(self, 'hw_constraints', None) is None:
            self.hw_constraints = state_dict.get('constraints', None)
        self.latest_schedule = state_dict['schedule']
        initial_state = state_dict['state']
        return initial_state

    def move(self):
        """Alter the current state, by changing at least one accelerator for its
           architectural parameters. The precision remains constant
           NOTE: This works for accelerators with the attribute 'precision'
        """
        new_state = self.state
        while new_state == self.state:
            new_state = [
                self.design_space.sample(
                    override_dict={'precision': self.accelerator_cfg.design_space['precision'][accelerator_idx]}
                )
                for accelerator_idx in range(self.num_accelerators)
            ]
        self.state = new_state

        logger.info("=> Move taken. New state:")
        for state in new_state:
            logger.info(f"\t{state}")

    def energy(self, initial=False):
        """Wrapper function for estimating the SA energy
        """
        start = time()
        logger.info(f"=> Beginning {'initial ' if initial else ''}energy calculation")
        energy = self._energy_evaluation()
        logger.info(f"Completed energy evaluation in {time() - start:.3e}s")
        logger.info(f"SA Energy results:\n"
                    f"\tEnergy={self.latest_energy}\n"
                    f"\tLatency={self.latest_latency}\n"
                    f"\tArea={self.latest_area}")
        logger.info("*--------------*")
        return energy

    def _energy_evaluation(self):
        """Evaluate the fitness of the current state
        """
        def violated_deadline(schedule):
            deadline = getattr(getattr(self, 'hw_constraints', None), 'deadline', None)
            # the constraint is not violated if a deadline is not given
            return deadline is not None and \
                   any(end_timestamp >= deadline for end_timestamp in schedule.end_timestamp.values())

        def violated_area_constraint(area):
            return area > getattr(self, 'initial_area', 0.0) * \
                         (1 - getattr(getattr(self, 'hw_constraints', None), 'area', 1.0)) 

        def violated_accuracy_constraint(arch, precision):
            try:
                return self.accuracy_lut.loc[
                            (self.accuracy_lut['Network'] == arch) &
                            (self.accuracy_lut['QuantBits'] == precision)
                        ]['Valid'].iloc[0] == 0
            except IndexError:
                return True

        # metrics to be accumulated
        energy_dict = {}
        area_dict = {}
        latency_dict = {}

        # iterate over each accelerator
        for accelerator in self.state:
            logger.info(f"\tEvaluating on accelerator: {accelerator._asdict()}")
            # iterate over each DNN
            for arch in self.workload.dnns.keys():
                logger.info(f"\t\tEvaluating on DNN: {arch}")
                # check if this evaluation was executed before
                if (arch, accelerator) in self.energy_dict:
                    logger.info(f"\t\tSkipping evaluation: already estimated")
                    continue

                # check accuracy constraint
                if violated_accuracy_constraint(arch, accelerator.precision):
                    logger.info(f"\t\tSkipping evaluation: accuracy violation")
                    # Invalid scheduling mappings are marked with negative weight (latency)
                    self.energy_dict[(arch, accelerator)] = -1
                    self.latency_dict[(arch, accelerator)] = -1
                    self.area_dict[(arch, accelerator)] = -1
                    continue

                # adjust timeloop with the accelerator parameters
                self.timeloop_wrapper.adjust_architecture(accelerator)

                energy_dict[(arch, accelerator)] = 0
                latency_dict[(arch, accelerator)] = 0
                area_dict[(arch, accelerator)] = 0
                # iterate over each timeloop problem (layer) of the DNN
                for problem_name in self.timeloop_problems_per_dnn[arch]:
                    logger.debug(f"\t\t\tEvaluating layer/problem: {problem_name}")

                    # run timeloop and get results
                    self.timeloop_wrapper.run(problem_name)
                    results = self.timeloop_wrapper.get_results()
                    energy_dict[(arch, accelerator)] += results.energy
                    latency_dict[(arch, accelerator)] += results.cycles
                    area_dict[(arch, accelerator)] += results.area
                    logger.debug(f"\t\t\tLayer-wise results: energy={results.energy}, "
                                 f"latency={results.cycles}, area={results.area}")

                logger.debug(f"\t\tEvaluation results for {arch} on {accelerator}:\n"
                             f"\t\t\tEnergy={energy_dict[(arch, accelerator)]}\n"
                             f"\t\t\tLatency={latency_dict[(arch, accelerator)]}\n"
                             f"\t\t\tArea={area_dict[(arch, accelerator)]}")

            # update stored metrics with executed evaluations
            self.energy_dict.update(energy_dict)
            self.latency_dict.update(latency_dict)
            self.area_dict.update(area_dict)

        logger.info("Completed mapping evaluation")

        # perform the scheduling and get a concrete DNN-to-accelerator mapping
        start = time()
        schedule = self.scheduler.run(items=list(self.workload.dnns.keys()),
                                      bins=self.state,
                                      cost_dict=self.energy_dict,
                                      weight_dict=self.latency_dict)
        # save schedule of latest move
        self.latest_schedule = schedule
        logger.debug(f"Schedule created in {time() - start:.3e}s")

        if schedule is None:
            # return in case of invalid schedule
            self.latest_energy = self.latest_latency = self.latest_area = None
            logger.info(f"Could not find valid schedule")
            return self.WORST_ENERGY

        # get results for energy, latency and area based on the final schedule
        self.latest_energy = sum([
            self.energy_dict[(entry.tag, entry.bin)] for entry in schedule.entries
        ])
        self.latest_latency = max([
            sum([
                self.latency_dict[(entry.tag, entry.bin)] for entry in entries
            ]) for bin, entries in schedule.as_dict(main_key='bin').items()
        ])
        self.latest_area = sum(
            self.area_dict[(entry.tag, entry.bin)] for entry in schedule.entries
        )

        # log the results of the scheduling
        schedule_str = '\n\t'.join([f'{entry.tag} -> {entry.bin}' for entry in schedule.entries])
        logger.info(f"Scheduler results:\n\t{schedule_str}")

        # save the results
        self.save_state()

        # check deadline and area constraints
        if violated_deadline(schedule) or \
           violated_area_constraint(self.latest_area):
            return self.WORST_ENERGY
        return self.latest_energy
