import logging
import os.path
from time import time
from src.accelerator_cfg import AcceleratorProfile
from src.timeloop import TimeloopWrapper
from src.scheduler import Scheduler
from src.optimizer import AcceleratorOptimizer


logger = logging.getLogger(__name__)


def run_baseline(args, workload, accuracy_lut):
    """Implementation of a baseline homogeneous accelerator
    """
    accel_cfg = AcceleratorProfile(args.accelerator_arch_type)

    # create the baseline multi-accelerator
    bl_evaluator = BaselineEvaluator(args, accel_cfg, workload, accuracy_lut)
    logger.info("Baseline accelerator:")
    for accelerator in bl_evaluator.state:
        logger.info(f"\t{accelerator}")

    # get the results of the evaluation
    bl_evaluator.evaluate()


class BaselineEvaluator(AcceleratorOptimizer):
    """Class to create and evaluate a multi-accelerator architecture which
       serves as our baseline (inheritance mostly for the timeloop wrapper)
    """
    def __init__(self,
                 args,
                 accelerator_cfg,
                 workload,
                 accuracy_lut
        ):
        self.accelerator_cfg = accelerator_cfg
        self.workload = workload
        self.accuracy_lut = accuracy_lut
        self.num_accelerators = args.baseline_num_accelerators
        self.logdir = args.logdir
        self.timeloop_wrapper = None
        self.timeloop_problems_per_dnn = None
        self.energy_dict = {}
        self.area_dict = {}
        self.latency_dict = {}
        self.latest_energy = self.latest_latency = self.latest_area = None
        self.latest_schedule = None

        # initialize accelerator
        self.init_accelerator(args)
        # initialize timeloop
        self.init_timeloop(args.layer_type_whitelist)
        # initialize scheduler
        self.scheduler = Scheduler(args.scheduler_type)

    def init_accelerator(self, args):
        """Create the multi-accelerator architecture
        """
        print(len(args.baseline_precision))
        if args.baseline_homogeneous:
            if len(args.baseline_precision) == 1:
                args.baseline_precision = [args.baseline_precision] * args.baseline_num_accelerators
        assert len(args.baseline_precision) == args.baseline_num_accelerators 

        self.state = []
        for accelerator_idx in range(args.baseline_num_accelerators):
            # get the value for each attribute of the accelerator
            values = [
                args.baseline_precision[accelerator_idx] 
                    if 'precision' in field.lower() else 
                getattr(self.accelerator_cfg, field)
                for field in self.accelerator_cfg.state._fields
            ]
            # create the accelerator instance
            self.state.append(self.accelerator_cfg.state(*values))

    def evaluate(self):
        """Evaluate the multi-accelerator architecture
        """
        start = time()
        logger.info("=> Beginning baseline evaluation")
        self._energy_evaluation()
        logger.info(f"Completed energy evaluation in {time() - start:.3e}s")
        logger.info(f"Baseline results:\n"
                    f"\tEnergy={self.latest_energy}\n"
                    f"\tLatency={self.latest_latency}\n"
                    f"\tArea={self.latest_area}")
        logger.info("*--------------*")

    def save_state(self):
        """Save the baseline measurements
        """
        savefile = os.path.join(self.logdir, 'baseline.results.pkl')
        super().save_state(save_state_to=savefile)
