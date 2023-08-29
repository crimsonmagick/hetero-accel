import torch
import logging
import numpy as np
import pytorch2timeloop as p2t
from copy import deepcopy
from src.net_wrapper import TorchNetworkWrapper
from src.models import create_model
from src.utils import weight_init, load_checkpoint
from src.rl.pruning import Pruner
from src.rl.quantization import Quantizer


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

    def init_model(self):
        """Overriding original method to include timeloop workload files
        """
        self.model = create_model(self.arch,
                                  self.dataset,
                                  self.batch_size,
                                  self.pretrained,
                                  parallel=not self.load_serialized,
                                  device_ids=self.gpus,
                                  verbose=True,
                                  to_timeloop=True)
        self.model.apply(weight_init)

        if self.resumed_checkpoint_path is not None:
            self.model, _ = load_checkpoint(
                self.model,
                self.resumed_checkpoint_path,
                model_device=self.device,
                to_cpu=self.device == 'cpu',
                #verbose=self.verbose
            )

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

    def compute_accelerator_statistics(self):
        # TODO: Use Timeloop/Accelergy to evaluate the accelerator metrics
        area = latency = power = energy = None
        return area, latency, power, energy

    def train(self, epochs):
        return super().train(epochs, self.train_loader, self.criterion, self.optimizer)

    def validate(self):
        return super().validate(self.valid_loader, self.criterion)

    def test(self):
        return super().test(self.test_loader, self.criterion)

