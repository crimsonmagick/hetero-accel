import yaml
import os
import shutil
import logging
import subprocess
import re
from time import time
from glob import glob
from collections import OrderedDict, namedtuple
from src import timeloop_dir
from src.accelerator_cfg import AcceleratorType


__all__ = ['TimeloopStats', 'TimeloopWrapper', 'TimeloopTemplate', 'TimeloopProblem']
logger = logging.getLogger(__name__)


TimeloopStats = namedtuple('TimeloopStats', ['gflops', 'utilization', 'cycles',
                                             'energy', 'edp', 'area'])


class TimeloopWrapper:
    """Wrapper for Timeloop+Accelergy tool
    """
    def __init__(self, accelerator_type):
        self.template = TimeloopTemplate(accelerator_type)
        self.workloads = OrderedDict()

    def init_files(self, timeloop_dir):
        """Initialize files and directories
        """
        os.makedirs(timeloop_dir, exist_ok=True)
        workload_dir = os.path.join(timeloop_dir, 'problem')
        os.makedirs(workload_dir, exist_ok=True)
        arch_dir = os.path.join(timeloop_dir, 'arch')
        os.makedirs(arch_dir, exist_ok=True)
        constraint_dir = os.path.join(timeloop_dir, 'constraints')
        os.makedirs(constraint_dir, exist_ok=True)
        output_dir = os.path.join(timeloop_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # copy files to the created directories
        shutil.copyfile(self.template.mapper, os.path.join(timeloop_dir, 'mapper.yaml'))
        for arch_file in self.template.arch:
            shutil.copy2(arch_file, arch_dir)
        for constraint_file in self.template.constraint:
            shutil.copy2(constraint_file, constraint_dir)

        # save files and directories for execution
        self.workload_dir = workload_dir
        self.arch_dir = arch_dir
        self.constraint_dir = constraint_dir
        self.output_dir = output_dir
        self.mapper_file = self.template.mapper

    def init_problem(self, problem_name, problem_type, dimensions, problem_filepath=None):
        """Initialize a workload wrapper
        """
        if problem_filepath is None:
            problem_filepath = os.path.join(self.workload_dir, f'{problem_name}_{problem_type}.yaml')
        problem = TimeloopProblem(problem_name, problem_type, dimensions, problem_filepath)
        problem.to_yaml()
        self.workloads[problem_name] = problem

    def run(self, problem_name):
        """Execute the Timeloop+Accelergy infrastructure via command-line
        """
        command = f'timeloop-mapper ' \
                  f'{self.arch_dir}/*.yaml ' \
                  f'{self.workloads[problem_name].problem_filepath} ' \
                  f'{self.mapper_file} ' \
                  f'{self.constraint_dir}/*.yaml ' \
                  #f'--outdir {self.output_dir} '
        logger.debug(f'timeloop-mapper command: {command}')
        start = time()
        p = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.debug(f"Executed timeloop-mapper command in {time() - start:.3e} "
                     f"with exitcode: {p.returncode}")

    def get_results(self, output_dir=None):
        """Get the results of a succesfull run from Timeloop. Note, there is a detailed script
           that does something similar:
           https://github.com/compstruct/procrustes-timeloop-model/blob/master/scripts/parse_timeloop_output.py#L16
        """
        if output_dir is None:
            assert self.output_dir is not None
            output_dir = self.output_dir

        stats_file = os.path.join(output_dir, 'timeloop-mapper.stats.txt')
        assert os.path.exists(stats_file)
        with open(stats_file, 'r') as f:
            stats = f.read()

        gflops = re.search('GFLOPs .*?: ([\d.]+)', stats).group(1)
        gflops = float(gflops)
        utilization = re.search('Utilization: ([\d.]+)', stats).group(1)
        utilization = float(utilization)
        cycles = re.search('Cycles: ([\d.]+)', stats).group(1)
        cycles = float(cycles)
        energy = re.search('Energy: ([\d.]+)', stats).group(1)
        energy = float(energy)
        edp = re.search('EDP.*?: (.*)', stats).group(1)
        edp = float(edp)
        area = re.search('Area: ([\d.]+)', stats).group(1)
        area = float(area)

        return TimeloopStats(gflops, utilization, cycles, energy, edp, area)

    # TODO: Search for ways to adjust timeloop according to weight pruning
    # TODO: Account for quantization; add custom MAC units to timeloop

    def adjust_precision(self, problem_name, precision):
        """Adjust the precision of the architecture
        """
        raise NotImplementedError

    def adjust_problem_dimension(self, problem_name, dimension, value=None, adjust_by=None):
        """Adjust the dimension size for the given workload-problem name
        """
        self.workloads[problem_name].adjust_dimension(dimension, value, adjust_by)
        self.workloads[problem_name].to_yaml()


class TimeloopTemplate:
    """Configuration environment for Timeloop files
    """
    def __init__(self, accelerator_type):
        if accelerator_type == AcceleratorType.Eyeriss:
            files_dir = os.path.join(timeloop_dir, '06-mapper-convlayer-eyeriss')
            self.mapper = os.path.join(files_dir, 'mapper', 'mapper.yaml')
            self.arch = [os.path.join(files_dir, 'arch', 'eyeriss_like.yaml')]
            self.arch += glob(os.path.join(files_dir, 'arch', 'components', '*.yaml'))
            self.constraint = glob(os.path.join(files_dir, 'constraints', '*.yaml'))
        else:
            raise NotImplementedError("Accelerator types other than "
                                      "Eyeriss-like are not supported")


class TimeloopProblem:
    """Utility class to handle and create a Timeloop-related workload
    """
    def __init__(self, name, problem_type, dimensions, problem_filepath):
        self.name = name
        self.dims = dimensions
        self.problem_filepath= problem_filepath
        self.config = None

        assert problem_type in ['Linear', 'Conv2d', 'AvgPool2d', 'MaxPool2d'], \
               f"Layer of type {problem_type} is not supported"
        self.problem_type = problem_type
        self._get_config()

    def _get_config(self):
        """Gather the configuration of the problem/workload in dict format
        """
        if self.problem_type in ['Conv2d', 'Linear']:
            self.config_conv_layer()
        elif 'pool' in self.problem_type.lower():
            self.config_pool_layer()

    def adjust_dimension(self, dimension, value=None, adjust_by=None):
        """Change/Adjust the value of a given workload dimension
        """
        if value is not None:
            self.config['instance'][dimension] = int(max(1, value))
        elif adjust_by is not None:
            self.config['instance'][dimension] = int(max(1, self.config['instance'][dimension] * adjust_by))
        else:
            raise ValueError("To change a workload dimension, specify either the absolute value or relative change")

    def to_yaml(self, filepath=None):
        """Create a yaml description of the workload
        """
        if self.config is None:
            self._get_config()

        if filepath is None:
            assert self.problem_filepath is not None
            filepath = self.problem_filepath
        with open(filepath, 'w') as f:
            f.write(yaml.dump({'problem': self.config}))

    def config_conv_layer(self):
        """Create the configuration for a Convolutional-type layer
        """
        self.dims['Q'] = int((self.dims['Xi'] - self.dims['S'] + 2 * self.dims['Wpad']) / self.dims['Wstr']) + 1
        self.dims['P'] = int((self.dims['Yi'] - self.dims['R'] + 2 * self.dims['Hpad']) / self.dims['Hstr']) + 1
        dimensions = ['C', 'M', 'R', 'S', 'N', 'P', 'Q']
        coefficients = ['Wstride', 'Hstride', 'Wdilation', 'Hdilation']

        config = {}
        config['shape'] = {}
        config['shape']['name'] = self.name
        config['shape']['dimensions'] = dimensions
        config['shape']['coefficients'] = [
            {
                'name': 'Hdilation',
                'default': 1
            },
            {
                'name': 'Wdilation',
                'default': 1
            },
            {
                'name': 'Hstride',
                'default': self.dims['Hstr']
            },
            {
                'name': 'Wstride',
                'default': self.dims['Wstr']
            }
        ]
        config['shape']['data-spaces'] = [
            {
                'name': 'Weights',
                'projection': [
                    [['M']],
                    [['C']],
                    [['R']],
                    [['S']]
                ]
            },
            {
                'name': 'Inputs',
                'projection': [
                    [['N']],
                    [['C']],
                    [['R', 'Wdilation'], ['P', 'Wstride']],
                    [['S', 'Hdilation'], ['Q', 'Hstride']],
                ]
            },
            {
                'name': 'Outputs',
                'projection': [
                    [['N']],
                    [['M']],
                    [['Q']],
                    [['P']]
                ],
                'read-write': True
            }
        ]
        config['instance'] = {
            'C': self.dims['C'],
            'M': self.dims['K'],
            'R': self.dims['R'],
            'S': self.dims['S'],
            'N': self.dims['N'],
            'P': self.dims['P'],
            'Q': self.dims['Q']
        }

        self.config = config

    def config_pool_layer(self):
        """Create the configuration for a Pooling layer
        """
        raise NotImplementedError

