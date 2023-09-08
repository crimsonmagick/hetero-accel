import os
import torch
import logging
import numpy as np
from types import SimpleNamespace
from glob import glob
from copy import deepcopy
from src import project_dir
from src.net_wrapper import TorchNetworkWrapper
from src.models import create_model
from src.utils import weight_init, load_checkpoint, get_sparsity
from src.accelerator_cfg import AcceleratorProfile
from src.compression.pruning import Pruner
from src.compression.quantization import Quantizer
from src.timeloop import TimeloopWrapper


logger = logging.getLogger(__name__)


class PruningQuantizationCompressor(TorchNetworkWrapper):
    """Class to handle all the compression-related actions
    """
    def __init__(self, args, data_loaders, model=None):
        super().__init__(args, model)

        self.original_model = deepcopy(self.model)
        self.train_loader, self.valid_loader, self.test_loader = data_loaders
        self.layers_to_compress = [name for name, module in self.model.named_modules()
                                   if isinstance(module, args.layer_type_whitelist)]
        logger.debug(f"Layers to compress: {self.layers_to_compress}")

        # pruner and quantizer for compression
        self.pruner = Pruner(args.pruning_group_type, self.layers_to_compress,
                             eridanus_window_w=self.accelerator_profile.width,
                             eridanus_window_h=self.accelerator_profile.height)
        self.quantizer = Quantizer(self.layers_to_compress)

        # timeloop wrapper to execute mapping searches and energy/area estimation
        self.timeloop_wrapper = TimeloopWrapper(self.accelerator_profile.type,
                                                os.path.join(self.logdir, f'{self.model.arch}_timeloop'))

        if self.model.is_image_classifier:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.model.device)
        else:
            raise NotImplementedError
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.01, weight_decay=1e-4)

    @classmethod
    def from_args(cls, args, data_loaders, model=None):
        compression_args = SimpleNamespace(logdir=args.logdir,
                                           pruning_high=args.pruning_high,
                                           pruning_low=args.pruning_low,
                                           quant_high=args.quant_high,
                                           quant_low=args.quant_low,
                                           layer_type_whitelist=(torch.nn.Conv2d,),
                                           pruning_group_type=args.pruning_group_type,
                                           accelerator_profile=AcceleratorProfile(args.accelerator_arch_type),
                                           # DNN args for inheritance from TorchNetworkWrapper
                                           profile_model=False,
                                           gpus=args.gpus,
                                           cpu=args.cpu,
                                           print_frequency=args.batch_print_frequency,
                                           verbose=args.model_verbose)
        return cls(compression_args, data_loaders, model)

    def reset(self):
        self.model = deepcopy(self.original_model)
        self.pruner.reset()
        self.quantizer.reset()

    def prune_and_quantize(self, pruning_ratio, quant_bits):
        assert self.pruning_low <= pruning_ratio <= self.pruning_high
        self.pruner.prune(self.model, pruning_ratio)
        assert self.quant_low <= quant_bits <= self.quant_high
        self.quantizer.quantize(self.model, quant_bits)

    def translate_pruning_action(self, pruning_action):
        pruning_action = pruning_action * (self.pruning_high - self.pruning_low) + self.pruning_low
        return np.round(pruning_action, 2).astype(float)

    def translate_quant_action(self, quant_action):
        quant_action = quant_action * (self.quant_high - self.quant_low) + self.quant_low
        return int(np.round(quant_action, 0))

    def compute_model_statistics(self):
        total_params = total_pruned = total_memory_size = 0
        for module_name, module in self.model.named_modules():
            if module_name not in self.layers_to_compress:
                continue

            bits = 32
            if hasattr(module, 'quant_metadata'):
                bits = module.quant_metadata.bits

            for param_name, param in module.named_parameters():
                if 'weight' not in param_name:
                    continue

                total_params += param.numel()
                total_pruned += param.abs().eq(0).sum()
                # memory size in bytes
                total_memory_size += (bits/8) * param.abs().gt(0).sum()

        sparsity = total_pruned / total_params
        return sparsity.item(), total_memory_size.item()

    def compute_accelerator_statistics(self, init=False):
        """Calculate the hardware-related efficiency metrics of the accelerator
           using Timeloop/Accelergy
        """
        total_area = total_latency = total_power = total_energy = 0

        if init:
            self.timeloop_wrapper.init_files()
            self.timeloop_wrapper.init_arch()

        # cleanup the timeloop files
        timeloop_dir = os.path.join(self.logdir, f'{self.model.arch}_timeloop')
        for timeloop_file in glob(os.path.join(project_dir, 'timeloop-mapper*')):
            os.remove(timeloop_file)

        # itearate over all layers
        layer_idx = 0
        for name, module in self.model.named_modules():
            if name not in self.layers_to_compress:
                continue

            # if specified, create the Timeloop workload files for each layer
            problem_filepath = os.path.join(self.timeloop_wrapper.workload_dir, f'layer{layer_idx}_{name}.yaml')
            if init:
                self.timeloop_wrapper.init_problem(name, self.summary[name].layer_type,
                                                   self.summary[name].dimensions, problem_filepath)

            # modifying the problem file to adjust for pruning
            sparsity_dict = get_sparsity(module.weight, True)
            self.timeloop_wrapper.adjust_problem_dimension(name, 'M',
                                                           adjust_by=(1 - sparsity_dict['filters']))
            self.timeloop_wrapper.adjust_problem_dimension(name, 'C',
                                                           adjust_by=(1 - sparsity_dict['channels']))

            # modifying the architecture to adjust for quantization
            bits = getattr(getattr(module, 'quant_metadata', None), 'bits', 32)
            self.timeloop_wrapper.adjust_precision(name, bits)
        
            # execute timeloop
            self.timeloop_wrapper.run(name)

            # gather results from simulation
            results = self.timeloop_wrapper.get_results(project_dir)
            logger.debug(f'Timeloop results for layer {layer_idx} ({name}): '
                         f'GFLOPS={results.gflops}, Utilization={results.utilization}, Cycles={results.cycles}, '
                         f'Energy={results.energy}, EDP={results.edp}, Area={results.area}')

            total_area += results.area
            total_latency += results.cycles
            total_energy += results.energy

            layer_idx += 1

        # cleanup the timeloop files
        for timeloop_file in glob(os.path.join(project_dir, 'timeloop-mapper*')):
            os.remove(timeloop_file)

        return total_area, total_latency, total_power, total_energy

    def train(self, epochs):
        return super().train(epochs, self.train_loader, self.criterion, self.optimizer)

    def validate(self):
        return super().validate(self.valid_loader, self.criterion)

    def test(self):
        return super().test(self.test_loader, self.criterion)

