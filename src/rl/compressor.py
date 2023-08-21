import torch
import logging
import numpy as np
from copy import deepcopy
from src.net_wrapper import TorchNetworkWrapper


logger = logging.getLogger(__name__)


class PruningQuantizationCompressor(TorchNetworkWrapper):
    """Class to handle all the compression-related actions
    """
    def __init__(self, model_args, data_loaders):
        super().__init__(model_args)

        self.original_model = deepcopy(self.model)
        self.train_loader, self.valid_loader, self.test_loader = data_loaders
        self.layers_to_compress = [name for name, module in self.model.named_modules()
                                   if isinstance(module, self.args.layer_type_whitelist)]
        # TODO: Automate this for rows or columns
        self.dim_to_prune = 1
        self.hook_handles = []
 
        if self.model.is_image_classifier:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.model.device)
        else:
            raise NotImplementedError
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.01, weight_decay=1e-4)

    def reset(self):
        self.model = deepcopy(self.original_model)
        # TODO: Unecessary with deepcopy?
        for handle in self.hook_handles:
            handle.remove()

    def prune_and_quantize(self, pruning_ratio, quant_bits):
        self.prune(pruning_ratio)
        self.quantize(quant_bits)

    def prune(self, pruning_ratio):
        """Prune on a given dimention of the weight matrix, globally, based on the pruning ratio
        """
        assert self.args.pruning_low <= pruning_ratio <= self.args.pruning_high

        norms = torch.tensor([], device=self.model.device)
        for name, module in self.model.named_modules():
            if name not in self.layers_to_compress:
                continue

            # convert weights to 2d to extract the per-layer norm values
            weights_2d = self.get_weights_2d(module)
            layer_norms = torch.norm(weights_2d, p=1, dim=self.dim_to_prune)
            # concatenate to global norm tensor
            norms = torch.cat((norms, layer_norms[torch.nonzero(layer_norms)]))

            logger.debug(f"Pruning {name}: original shape {module.weight.shape}, 2d {weights_2d.shape} "
                         f"norm {layer_norms.shape}")

        # find the threshold value for all norms
        threshold = torch.quantile(norms.squeeze(), pruning_ratio)
 
        for name, module in self.model.named_modules():
            if name not in self.layers_to_compress:
                continue

            weights_2d = self.get_weights_2d(module)
            layer_norms = torch.norm(weights_2d, p=1, dim=self.dim_to_prune)

            mask = layer_norms.gt(threshold).type(module.weight.data.type())
            mask = mask.expand(np.prod(module.weight.shape[1:]), module.weight.shape[0]).t()
            mask = mask.view(*module.weight.shape)

            logger.debug(f"Pruning {name}: original shape {module.weight.shape}, 2d {weights_2d.shape} "
                         f"")

            module.weight.data.mul_(mask) 

    def get_weights_2d(self, module):
        with torch.no_grad():
            weights_2d = module.weight.view(-1, np.prod(module.weight.shape[1:]))
        return weights_2d
            

    def quantize(self, quant_bits):
        # TODO: add hooks for quantization
        raise NotImplementedError

    def translate_pruning_action(self, pruning_action):
        pruning_action = pruning_action * (self.args.pruning_high - self.args.pruning_low) + self.args.pruning_low
        return np.round(pruning_action, 2).astype(float)

    def translate_quant_action(self, quant_action):
        quant_action = quant_action * (self.args.quant_high - self.args.quant_low) + self.args.quant_low
        return int(np.round(quant_action, 0))

    def compute_model_statistics(self):
        total_params = total_pruned = total_memory_size = 0
        for module_name, module in self.model.named_modules():
            bits = getattr(module, 'weight_bits', 32)
            for param_name, param in module.named_parameters():
                total_params += param.numel()
                total_pruned += param.eq(0).sum()
                # memory size in bytes
                total_memory_size += (bits/8) * param.gt(0).sum()

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

