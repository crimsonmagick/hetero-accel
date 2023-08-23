import logging
import torch
import numpy as np
from enum import Enum


logger = logging.getLogger(__name__)


class PruningGroupType(Enum):
    Rows = 1
    Columns = 2
    Filters = 3
    Channels = 4
    Weights = 5
    EridanusBlocks = 6


class Pruner:
    """Class to handle pruning of a DNN model. Multiple types of pruning are supported
    """
    def __init__(self, args, layers_to_compress):
        self.layers_to_compress = layers_to_compress
        # configure pruning function based on the given group type
        if args.pruning_group_type in (PruningGroupType.Rows, PruningGroupType.Columns):
            self.prune = self.prune_rows_columns
            self.dim_to_prune = 2 if args.pruning_group_type == PruningGroupType.Rows else 1
        elif args.pruning_group_type == PruningGroupType.Filters:
            self.prune = self.prune_filters
        elif args.pruning_group_type == PruningGroupType.Channels:
            self.prune = self.prune_channels
        elif args.pruning_group_type == PruningGroupType.Weights:
            self.prune = self.prune_weights
        elif args.pruning_group_type == PruningGroupType.EridanusBlocks:
            self.prune = self.prune_eridanus_blocks

    def prune_rows_columns(self, model, pruning_ratio):
        """Prune on a given dimention (columns or rows) of the weight matrix, globally,
           based on the pruning ratio.
           NOTE: Pruning columns with the method below is the same as pruning filters
        """
        def get_weights_2d(module):
            with torch.no_grad():
               return module.weight.view(-1, np.prod(module.weight.shape[1:]))

        norms = torch.tensor([], device=model.device)
        for name, module in model.named_modules():
            if name not in self.layers_to_compress:
                continue

            # convert weights to 2d to extract the per-layer norm values
            weights_2d = get_weights_2d(module)
            layer_norms = torch.norm(weights_2d, p=1, dim=self.dim_to_prune)
            # concatenate to global norm tensor
            norms = torch.cat((norms, layer_norms[torch.nonzero(layer_norms)]))

        # find the threshold value for all norms
        threshold = torch.quantile(norms.squeeze(), pruning_ratio)
        logger.debug(f"Pruning ratio {pruning_ratio} turned to threshold {threshold:.3e}")
 
        for name, module in model.named_modules():
            if name not in self.layers_to_compress:
                continue

            weights_2d = get_weights_2d(module)
            layer_norms = torch.norm(weights_2d, p=1, dim=self.dim_to_prune)

            mask = layer_norms.gt(threshold).type(module.weight.data.type())
            mask = mask.expand(np.prod(module.weight.shape[1:]), module.weight.shape[0]).t()
            mask = mask.view(*module.weight.shape)
            # apply the pruning to the layer
            module.weight.data.mul_(mask) 

    def prune_filters(self, model, pruning_ratio):
        """Prune filters globally based on the given pruning ratio
        """
        norms = torch.tensor([], device=model.device)
        for name, module in model.named_modules():
            if name not in self.layers_to_compress:
                continue
            
            with torch.no_grad():
                filter_norms = torch.norm(module.weight.view(-1, np.prod(module.weight.shape[1:])), p=1, dim=1)
            # concatenate to global norm tensor
            norms = torch.cat((norms, filter_norms[torch.nonzero(filter_norms)]))

        # find the threshold value for all norms
        threshold = torch.quantile(norms.squeeze(), pruning_ratio)
        logger.debug(f"Pruning ratio {pruning_ratio} turned to threshold {threshold:.3e}")
 
        for name, module in model.named_modules():
            if name not in self.layers_to_compress:
                continue

            with torch.no_grad():
                filter_norms = torch.norm(module.weight.view(-1, np.prod(module.weight.shape[1:])), p=1, dim=1)

            mask = filter_norms.gt(threshold).type(module.weight.data.type())
            mask = mask.expand(np.prod(module.weight.shape[1:]), module.weight.shape[0]).t()
            mask = mask.view(*module.weight.shape)
            # apply the pruning to the layer
            module.weight.data.mul_(mask) 

    def prune_channels(self, model, pruning_ratio):
        """Prune channels globally based on the given pruning ratio
        """
        norms = torch.tensor([], device=model.device)
        for name, module in model.named_modules():
            if name not in self.layers_to_compress:
                continue
            
            with torch.no_grad():
                weights = module.weight.transpose(0, 1).contiguous()
                channel_norms = torch.norm(weights.view(-1, np.prod(weights.shape[1:])), p=1, dim=1)
            # concatenate to global norm tensor
            norms = torch.cat((norms, channel_norms[torch.nonzero(channel_norms)]))

        # find the threshold value for all norms
        threshold = torch.quantile(norms.squeeze(), pruning_ratio)
        logger.debug(f"Pruning ratio {pruning_ratio} turned to threshold {threshold:.3e}")
 
        for name, module in model.named_modules():
            if name not in self.layers_to_compress:
                continue

            with torch.no_grad():
                weights = module.weight.transpose(0, 1).contiguous()
                channel_norms = torch.norm(weights.view(-1, np.prod(weights.shape[1:])), p=1, dim=1)

            mask = channel_norms.gt(threshold).type(module.weight.data.type())
            mask = mask.expand(module.weight.shape[0], module.weight.shape[1]).unsqueeze(-1)
            mask = mask.expand(module.weight.shape[0],
                               module.weight.shape[1],
                               module.weight.shape[2] * module.weight.shape[3])
            mask = mask.view(*module.weight.shape)
            # apply the pruning to the layer
            module.weight.data.mul_(mask) 

    def prune_weights(self, model, pruning_ratio):
        """Prune weights globally based on the given pruning ratio
        """
        all_weights = torch.tensor([], device=model.device)
        for name, module in model.named_modules():
            if name not in self.layers_to_compress:
                continue
            with torch.no_grad():
                weights = module.weight.abs().flatten()
            all_weights = torch.cat((all_weights, weights[torch.nonzero(weights)]))

        # find the threshold value for all individual weights
        threshold = torch.quantile(all_weights.squeeze(), pruning_ratio)
        logger.debug(f"Pruning ratio {pruning_ratio} turned to threshold {threshold:.3e}")
 
        for name, module in model.named_modules():
            if name not in self.layers_to_compress:
                continue
            mask = module.weight.abs().gt(threshold).type(module.weight.data.type())
            # apply the pruning to the layer
            module.weight.data.mul_(mask) 

    def prune_eridanus_blocks(self, model, pruning_ratio):
        """Prune Eridanus-like blocks globally based on the given pruning ratio
        """
        raise NotImplementedError

    def reset(self):
        pass

