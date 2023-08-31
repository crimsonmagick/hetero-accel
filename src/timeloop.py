import yaml
import os
import logging
from glob import glob
from src import timeloop_dir
from src.args import AcceleratorType


logger = logging.getLogger(__name__)


class TimeloopConfig:
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
    def __init__(self, name, dimensions, problem_type):
        self.name = name
        self.dims = dimensions

        assert problem_type in ['Linear', 'Conv2d', 'AvgPool2d', 'MaxPool2d'], \
               f"Layer of type {problem_type} is not supported"
        self.problem_type = problem_type

    def to_yaml(self, filepath=None):
        """Create a yaml description of the workload
        """
        if self.problem_type in ['Conv2d', 'Linear']:
            config = self.config_conv_layer()
        elif 'pool' in self.problem_type.lower():
            config = self.config_pool_layer()

        if filepath is not None:
            with open(filepath, 'w') as f:
                f.write(yaml.dump(config))
        return config

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
        return {'problem': config}

    def config_pool_layer(self):
        """Create the configuration for a Pooling layer
        """

        raise NotImplementedError

