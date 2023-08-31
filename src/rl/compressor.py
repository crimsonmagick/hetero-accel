import re
import os
import shutil
import yaml
import torch
import logging
import subprocess
import numpy as np
from time import time
from glob import glob
from copy import deepcopy
from src import project_dir
from src.net_wrapper import TorchNetworkWrapper
from src.rl.pruning import Pruner
from src.rl.quantization import Quantizer
from src.timeloop import TimeloopProblem


logger = logging.getLogger(__name__)


class PruningQuantizationCompressor(TorchNetworkWrapper):
    """Class to handle all the compression-related actions
    """
    def __init__(self, args, data_loaders):
        super().__init__(args)

        self.original_model = deepcopy(self.model)
        self.train_loader, self.valid_loader, self.test_loader = data_loaders
        self.layers_to_compress = [name for name, module in self.model.named_modules()
                                   if isinstance(module, args.layer_type_whitelist)]
        self.pruning_high = args.pruning_high
        self.pruning_low = args.pruning_low
        self.quant_high = args.quant_high
        self.quant_low = args.quant_low

        self.pruner = Pruner(args, self.layers_to_compress)
        self.quantizer = Quantizer(args, self.layers_to_compress)

        if self.model.is_image_classifier:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.model.device)
        else:
            raise NotImplementedError
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.01, weight_decay=1e-4)

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

            bits = getattr(module, 'weight_bits', 32)
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

        # directories and files for timeloop
        timeloop_dir = os.path.join(self.logdir, f'{self.model.arch}_timeloop')
        workload_dir = os.path.join(timeloop_dir, 'problem')
        mapper_file = os.path.join(timeloop_dir, 'mapper.yaml')
        arch_dir = os.path.join(timeloop_dir, 'arch')
        constraint_dir = os.path.join(timeloop_dir, 'constraints')
        outdir = os.path.join(timeloop_dir, 'output')
        if init and not os.path.isdir(timeloop_dir):
            os.makedirs(timeloop_dir)
            os.makedirs(workload_dir)
            os.makedirs(arch_dir)
            os.makedirs(constraint_dir)
            os.makedirs(outdir)
            shutil.copyfile(self.timeloop_files.mapper, mapper_file)
            for arch_file in self.timeloop_files.arch:
                shutil.copy2(arch_file, arch_dir)
            for constraint_file in self.timeloop_files.constraint:
                shutil.copy2(constraint_file, constraint_dir)

        # cleanup the timeloop files
        for timeloop_file in glob(os.path.join(project_dir, 'timeloop-mapper*')):
            os.remove(timeloop_file)

        # itearate over all layers
        layer_idx = 0
        for name, module in self.model.named_modules():
            if name not in self.layers_to_compress:
                continue

            # if specified, create the Timeloop workload files for each layer
            this_outdir = os.path.join(outdir, f'layer{layer_idx}_{name}')
            problem_filepath = os.path.join(workload_dir, f'layer{layer_idx}_{name}.yaml')
            if init:
                os.makedirs(this_outdir)
                dims = self.summary[name].dimensions
                layer_type = self.summary[name].layer_type
                TimeloopProblem(name, dims, layer_type).to_yaml(problem_filepath)

            # execute the Timeloop/Accelergy infrastructure
            command = f'timeloop-mapper ' \
                      f'{arch_dir}/*.yaml ' \
                      f'{problem_filepath} ' \
                      f'{mapper_file} ' \
                      f'{constraint_dir}/*.yaml ' \
                      #f'--outdir {this_outdir} '
            logger.debug(f'timeloop-mapper command: {command}')
            start = time()
            p = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            logger.debug(f"Executed timeloop-mapper command in {time() - start:.3e} "
                         f"with exitcode: {p.returncode}")

            # accumulate produced metrics
            stats_file = os.path.join(project_dir, 'timeloop-mapper.stats.txt')
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
            logger.debug(f'Timeloop results for layer {layer_idx} ({name}): '
                         f'GFLOPS={gflops}, Utilization={utilization}, Cycles={cycles},'
                         f'Energy={energy}, EDP={edp}, Area={area}')

            total_area += area
            total_latency += cycles
            total_energy += energy

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

