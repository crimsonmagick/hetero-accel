import yaml
import os
import shutil
import logging
import subprocess
import re
from time import time
from copy import deepcopy
from glob import glob
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from src import timeloop_dir
from src.accelerator_cfg import AcceleratorType


__all__ = ['TimeloopStats', 'TimeloopWrapper', 'TimeloopTemplate', 'TimeloopProblem', 'TimeloopArch', 'TimeloopMapper']

logger = logging.getLogger(__name__)

TIMELOOP_VERSION = 0.3


TimeloopStats = namedtuple('TimeloopStats', ['gflops', 'utilization', 'cycles',
                                             'energy', 'edp', 'area'])


class TimeloopWrapper:
    """Wrapper for Timeloop+Accelergy tool
    """
    def __init__(self, accelerator_type, workdir, workloads=None):
        self.template = TimeloopTemplate(accelerator_type)
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)
        self.workloads = workloads if workloads is not None else OrderedDict()
        self.init_files()
        self.init_arch(accelerator_type)
        self.init_mapper()

    def init_files(self):
        """Initialize files and directories
        """
        workload_dir = os.path.join(self.workdir, 'problem')
        os.makedirs(workload_dir, exist_ok=True)
        constraint_dir = os.path.join(self.workdir, 'constraints')
        os.makedirs(constraint_dir, exist_ok=True)
        output_dir = os.path.join(self.workdir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # copy constraint files to the created directory
        for constraint_file in self.template.constraint:
            shutil.copy2(constraint_file, constraint_dir)

        # save files and directories for execution
        self.workload_dir = workload_dir
        self.constraint_dir = constraint_dir
        self.output_dir = output_dir

    def init_problem(self, problem_name, problem_type, dimensions, problem_filepath=None):
        """Initialize a workload wrapper
        """
        if problem_name in self.workloads:
            return

        if problem_filepath is None:
            problem_filepath = os.path.join(self.workload_dir, f'{problem_name}_{problem_type}.yaml')

        problem = TimeloopProblem(problem_name, problem_type, dimensions, problem_filepath)
        problem.to_yaml()
        self.workloads[problem_name] = problem

    def init_arch(self, accelerator_type):
        """Initialize the accelerator architecture description
        """
        arch_dir = os.path.join(self.workdir, 'arch')
        os.makedirs(arch_dir, exist_ok=True)
        for arch_file in self.template.arch:
            shutil.copy2(arch_file, arch_dir)

        arch_file = os.path.join(arch_dir, os.path.basename(self.template.arch[0]))
        component_files = [os.path.join(arch_dir, os.path.basename(compfile))
                           for compfile in self.template.arch[1:]]
        self.arch = TimeloopArch(accelerator_type, arch_dir, arch_file, component_files)
 
    def init_mapper(self):
        """Initialize the mapper object for the mapping optimization
        """
        mapper_file = os.path.join(self.workdir, 'mapper.yaml')
        shutil.copyfile(self.template.mapper, mapper_file)
        self.mapper = TimeloopMapper(mapper_file)

    def run(self, problem_name):
        """Execute the Timeloop+Accelergy infrastructure via command-line
        """
        command = f'timeloop-mapper ' \
                  f'{self.arch.arch_filepath} ' \
                  f'{" ".join(self.arch.component_files)} ' \
                  f'{self.workloads[problem_name].problem_filepath} ' \
                  f'{self.mapper.mapper_filepath} ' \
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
        gflops = float(gflops)              # @1GHz
        utilization = re.search('Utilization: ([\d.]+)', stats).group(1)
        utilization = float(utilization)    # non-unit
        cycles = re.search('Cycles: ([\d.]+)', stats).group(1)
        cycles = float(cycles)              # non-unit
        energy = re.search('Energy: ([\d.]+)', stats).group(1)
        energy = float(energy)              # uJ
        edp = re.search('EDP.*?: (.*)', stats).group(1)
        edp = float(edp)                    # J * cycle
        # TODO: Area comes back as 0.0. Maybe because of DRAM area=0.0. Fix.
        area = re.search('Area: ([\d.]+)', stats).group(1)
        area = float(area)                  # mm^2

        return TimeloopStats(gflops, utilization, cycles, energy, edp, area)

    # TODO: Search for ways to adjust timeloop according to weight pruning

    def adjust_precision(self, precision):
        """Adjust the precision of the architecture, including arithmetic and memory units
        """
        self.arch.adjust_precision(precision)
        self.arch.to_yaml()

    def adjust_problem_dimension(self, problem_name, dimension, value=None, adjust_by=None):
        """Adjust the dimension size for the given workload-problem name
        """
        self.workloads[problem_name].adjust_dimension(dimension, value, adjust_by)
        self.workloads[problem_name].to_yaml()

    def adjust_pe_array(self, pe_array_x, pe_array_y):
        """Adjust the dimensions of the PE array
        """
        self.arch.adjust_pe_array(pe_array_x, pe_array_y)
        self.arch.to_yaml()

    def adjust_memories(self, accelerator_instance):
        """Adjust the accelerator-specific memories. Note, the accelerator
           instance is given, instead of explict parameters
        """
        self.arch.adjust_memories(accelerator_instance)
        self.arch.to_yaml()

    def adjust_mapper(self, param_name, param_value):
        self.mapper.adjust_param(param_name, param_value)
        self.mapper.to_yaml()


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
            raise NotImplementedError("Accelerator types other than Eyeriss-like are not supported")


class TimeloopProblem:
    """Utility class to handle and create a Timeloop-related workload
    """
    def __init__(self, name, problem_type, dimensions, problem_filepath):
        self.name = name
        self.dims = dimensions
        self.problem_filepath = problem_filepath
        self.config = None

        assert problem_type in ['Linear', 'Conv2d', 'AvgPool2d', 'MaxPool2d'], \
               f"Layer of type {problem_type} is not supported"
        self.problem_type = problem_type
        self.get_config()

    def get_config(self):
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
            self.get_config()

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


class TimeloopArch:
    """Utility class to handle the architectural parameters of the accelerator
       when using timeloop
    """
    def __init__(self, accelerator_type, workdir, arch_file, component_files):
        self.workdir = workdir
        self.arch_filepath = arch_file
        self.component_files = component_files 
        self.name = self.__class__.__name__

        # functions for specific accelerator types
        if accelerator_type == AcceleratorType.Eyeriss:
            self.adjust_precision = self._adjust_precision_eyeriss
            self.adjust_memories = self._adjust_memories_eyeriss
            self.get_default_params = self._get_default_params_eyeriss
            self.get_config = self._get_config_eyeriss

        else:
            raise NotImplementedError("Accelerator types other than Eyeriss-like are not supported")

        # initialize dict with parameters
        self.get_default_params()
        self.init_params = deepcopy(self.params)
        self.get_config()

    def to_yaml(self, filepath=None):
        """Write the configuration of the architecture to a yaml file
        """
        if self.config is None:
            self.get_config()

        if filepath is None:
            assert self.arch_filepath is not None
            filepath = self.arch_filepath
        with open(filepath, 'w') as f:
            f.write(yaml.dump({'architecture': self.config}))
    
    def adjust_params(self, params):
        """Generic function to override parameter values
        """
        for param_name, value in params.items():
            assert hasattr(self.params, param_name), f'{param_name} is not a valid parameter'
            setattr(self.params, param_name, value)
        # update the configuration with new parameters
        self.get_config()

    def adjust_pe_array(self, pe_x, pe_y):
        """Adjust the dimensions of the PE array
        """
        params = {'pe_array_x': pe_x,
                  'pe_array_y': pe_y}
        self.adjust_params(params)

    ### Accelerator-specific functions ###

    def _adjust_precision_eyeriss(self, precision):
        """Adjust the data precision of the architecture, including memory and compute units
        """
        # NOTE: For each memory unit, the following assertion must be satisfied:
        #       width % (word_bits * block_size) == 0
        #       Also, another assertion (cluster_size is probably width/word_bits):
        #       specs_.instances.Get() % specs_.cluster_size.Get() == 0

        def adjust_mem_width(width, word_bits_precision):
            """This is the technique for adjusting the memory width for
               specific word bits precision. Three options are available:
               1) Leave to baseline (16 bits): Causes errors because of 
                  assertion for division (first one above)
               2) Match the width with the word bits. This works, but creates
                  cluster sizes of 1, which do not lead to low energy.
               3) Round to next higher divisor of the word bits. This gives
                  the best energy/area trade-off (i.e., lower energy but not too
                  high area). However, the cluster size assertion must be satisfied,
                  so needs careful handling.
               We go for the third option here. It also was tested with [2, 8] bits
               and timeloop does not throw any errors
            """
            return width + (width % word_bits_precision)

        # We only change the parameters of the MAC unit and the scratchpads.
        # The DRAM and SRAM are not affected by the precision of the MAC units,
        # and any changes to the dummy register buffers do not affect the results 
        params = {
            'mac_datawidth': precision,
            'mac_class': 'fpmac' if precision == 32 else 'intmac',
            'ifmap_spad_word_bits': precision,
            'weights_spad_word_bits': precision,
            'psum_spad_word_bits': precision,
            'ifmap_spad_width': adjust_mem_width(self.init_params.ifmap_spad_width, precision),
            'weights_spad_width': adjust_mem_width(self.init_params.weights_spad_width, precision),
            'psum_spad_width': adjust_mem_width(self.init_params.psum_spad_width, precision)
        }
        self.adjust_params(params)

    def _adjust_memories_eyeriss(self, accelerator_instance):
        """Adjust each specific memory unit of the accelerator
        """
        params = {
            'sram_depth': accelerator_instance.sram_size,
            'ifmap_spad_depth': accelerator_instance.ifmap_spad_size,
            'weights_spad_depth': accelerator_instance.weights_spad_size,
            'psum_spad_depth': accelerator_instance.psum_spad_size
        }
        self.adjust_params(params)

    def _get_default_params_eyeriss(self):
        """Get the default parameters for all levels of an
           Eyeriss-like architecture
        """
        self.params = SimpleNamespace()
        self.params.pe_array_x = 14
        self.params.pe_array_y = 16

        self.params.technology = '45nm'
        # external DRAM attributes
        self.params.dram_width = 64
        self.params.dram_word_bits = 16
        # global SRAM attributes
        self.params.sram_class = 'smartbuffer_SRAM'
        self.params.sram_depth = 16384
        self.params.sram_width = 64
        self.params.sram_n_banks = 32
        self.params.sram_word_bits = 16
        self.params.sram_read_bandwidth = 16
        self.params.sram_write_bandwidth = 16
        # dummy register file attributes
        self.params.regfile_depth = 16
        self.params.regfile_width = 16
        self.params.regfile_word_bits = 16
        # class for implementing scratchpads
        self.params.spad_class = 'smartbuffer_RF'
        # attributes for IFM scratchpad
        self.params.ifmap_spad_depth = 12
        self.params.ifmap_spad_width = 16
        self.params.ifmap_spad_word_bits = 16
        self.params.ifmap_spad_read_bandwidth = 2
        self.params.ifmap_spad_write_bandwidth = 2
        # attributes for Weights' scratchpad
        self.params.weights_spad_depth = 192
        self.params.weights_spad_width = 16
        self.params.weights_spad_word_bits = 16
        self.params.weights_spad_read_bandwidth = 2
        self.params.weights_spad_write_bandwidth = 2
        # attributes for Partial Sums' scratchpad
        self.params.psum_spad_depth = 16
        self.params.psum_spad_width = 16
        self.params.psum_spad_update_fifo_depth = 2
        self.params.psum_spad_word_bits = 16
        self.params.psum_spad_read_bandwidth = 2
        self.params.psum_spad_write_bandwidth = 2
        # MAC unit attributes
        self.params.mac_class = 'intmac'
        self.params.mac_datawidth = 16

    def _get_config_eyeriss(self):
        """Write the architectural description of an Eyeriss-like
           architecture in a dict format
        """
        config = {}
        config['version'] = TIMELOOP_VERSION

        level1 = {}
        level1['name'] = 'system'
        level1['local'] = [
            {
                'name': 'DRAM',
                'class': 'DRAM',
                'attributes': {
                    'type': 'LPDDR4',
                    'width': self.params.dram_width,
                    'block-size': self.params.dram_width // self.params.dram_word_bits,
                    'word-bits': self.params.dram_word_bits
                }
            }
        ]

        level2 = {}
        level2['name'] = 'eyeriss'
        level2['attributes'] = {
            'technology': self.params.technology
        }
        level2['local'] = [
            {
                'name': 'shared_glb',
                'class': self.params.sram_class,
                'attributes': {
                    'memory_depth': self.params.sram_depth,
                    'memory_width': self.params.sram_width,
                    'n_banks': self.params.sram_n_banks,
                    'block-size': self.params.sram_width // self.params.sram_word_bits,
                    'word-bits': self.params.sram_word_bits,
                    'read_bandwidth': self.params.sram_read_bandwidth,
                    'write_bandwidth': self.params.sram_write_bandwidth
                }
            },
            {
                'name': f'DummyBuffer[0..{self.params.pe_array_x - 1}]',
                'class': 'regfile',
                'attributes': {
                    'depth': self.params.regfile_depth,
                    'width': self.params.regfile_width,
                    'word-bits': self.params.regfile_word_bits,
                    'block-size': self.params.regfile_width // self.params.regfile_word_bits,
                    'meshX': self.params.pe_array_x
                }
            }
        ]

        level3 = {'name': f'PE[0..{self.params.pe_array_x * self.params.pe_array_y - 1}]'}
        level3_ifmap = {
            'name': 'ifmap_spad',
            'class': self.params.spad_class,
            'attributes': {
                'memory_depth': self.params.ifmap_spad_depth,
                'memory_width': self.params.ifmap_spad_width,
                'block-size': self.params.ifmap_spad_width // self.params.ifmap_spad_word_bits,
                'word-bits': self.params.ifmap_spad_word_bits,
                'meshX': self.params.pe_array_x,
                'read_bandwidth': self.params.ifmap_spad_read_bandwidth,
                'write_bandwidth': self.params.ifmap_spad_write_bandwidth,
            }
        }
        level3_weights = {
            'name': 'weights_spad',
            'class': self.params.spad_class,
            'attributes': {
                'memory_depth': self.params.weights_spad_depth,
                'memory_width': self.params.weights_spad_width,
                'block-size': self.params.weights_spad_width // self.params.weights_spad_word_bits,
                'word-bits': self.params.weights_spad_word_bits,
                'meshX': self.params.pe_array_x,
                'read_bandwidth': self.params.weights_spad_read_bandwidth,
                'write_bandwidth': self.params.weights_spad_write_bandwidth,
            }
        }
        level3_psum = {
            'name': 'psum_spad',
            'class': self.params.spad_class,
            'attributes': {
                'memory_depth': self.params.psum_spad_depth,
                'memory_width': self.params.psum_spad_width,
                'update_fifo_depth': self.params.psum_spad_update_fifo_depth,
                'block-size': self.params.psum_spad_width // self.params.psum_spad_word_bits,
                'word-bits': self.params.psum_spad_word_bits,
                'meshX': self.params.pe_array_x,
                'read_bandwidth': self.params.psum_spad_read_bandwidth,
                'write_bandwidth': self.params.psum_spad_write_bandwidth,
            }
        }
        level3_mac = {
            'name': 'mac',
            'class': self.params.mac_class,
            'attributes': {
                'datawidth': self.params.mac_datawidth,
                'meshX': self.params.pe_array_x
            }
        }
        level3['local'] = [level3_ifmap, level3_weights, level3_psum, level3_mac]

        level2['subtree'] = [level3]
        level1['subtree'] = [level2]
        config['subtree'] = [level1]

        self.config = config


class TimeloopMapper:
    """Utility wrapper class fot the mapping optimizer
    """
    def __init__(self, mapper_file):
        self.mapper_filepath = mapper_file
        self.get_params()
        self.get_config()

    def get_params(self):
        """Collect the default configuration parameters of the mapper 
        """
        self.params = SimpleNamespace()
        self.params.optimization_metrics = ['delay', 'energy']
        self.params.live_status = False
        self.params.num_threads = 8
        self.params.timeout = 15000
        self.params.victory_condition = 500
        self.params.algorithm = 'random-pruned'
        self.params.max_permutations_per_if_visit = 16

    def get_config(self):
        """Write the configuration parameters to a yaml-like dict
        """
        config = {
            'optimization-metrics': self.params.optimization_metrics,
            'live-status': self.params.live_status,
            'num-threads': self.params.num_threads,
            'timeout': self.params.timeout,
            'victory-condition': self.params.victory_condition,
            'algorithm': self.params.algorithm,
            'max-permutations-per-if-visit': self.params.max_permutations_per_if_visit
        }
        self.config = config

    def adjust_param(self, param_name, value):
        """Generic function to override a parameter value
        """
        assert hasattr(self.params, param_name), f'{param_name} is not a valid parameter'
        setattr(self.params, param_name, value)
        # update the configuration with new parameters
        self.get_config()

    def to_yaml(self, filepath=None):
        """Write the configuration of the mapper to a yaml file
        """
        if self.config is None:
            self.get_config()

        if filepath is None:
            assert self.mapper_filepath is not None
            filepath = self.mapper_filepath

        with open(filepath, 'w') as f:
            f.write(yaml.dump({'mapper': self.config}))
