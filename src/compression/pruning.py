import logging
import torch
import numpy as np
from enum import Enum
from types import SimpleNamespace


logger = logging.getLogger(__name__)


class PruningGroupType(Enum):
    Rows = 1
    Columns = 2
    Filters = 3
    Channels = 4
    Kernels = 5
    Weights = 6
    EridanusBlocks = 7


class Pruner:
    """Class to handle pruning of a DNN model. Multiple types of pruning are supported
    """
    def __init__(self, pruning_group_type, layers_to_compress, **kwargs):
        self.layers_to_compress = layers_to_compress
        self.args = SimpleNamespace(**kwargs)

        # configure pruning function based on the given group type
        if pruning_group_type in (PruningGroupType.Rows, PruningGroupType.Columns):
            self.prune = self.prune_rows_columns
            self.dim_to_prune = 0 if pruning_group_type == PruningGroupType.Rows else 1
        elif pruning_group_type == PruningGroupType.Filters:
            self.prune = self.prune_filters
        elif pruning_group_type == PruningGroupType.Channels:
            self.prune = self.prune_channels
        elif pruning_group_type == PruningGroupType.Kernels:
            self.prune = self.prune_kernels
        elif pruning_group_type == PruningGroupType.Weights:
            self.prune = self.prune_weights
        elif pruning_group_type == PruningGroupType.EridanusBlocks:
            self.prune = self.prune_eridanus_blocks

    def prune_rows_columns(self, model, pruning_ratio):
        """Prune on a given dimension (columns or rows) of the weight matrix, globally,
           based on the pruning ratio.
           NOTE: Pruning columns with the method below is the same as pruning filters
        """
        def get_weights_2d(module):
            with torch.no_grad():
               return module.weight.view(-1, np.prod(module.weight.shape[1:]))

        norms = torch.tensor([], device=model.device, requires_grad=False)
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
            setattr(module, 'pruning_metadata', {'mask': mask})

    def prune_filters(self, model, pruning_ratio):
        """Prune filters globally based on the given pruning ratio
        """
        norms = torch.tensor([], device=model.device, requires_grad=False)
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

            # only works for 4-D weight tensors
            if module.weight.dim() != 4:
                continue

            with torch.no_grad():
                filter_norms = torch.norm(module.weight.view(-1, np.prod(module.weight.shape[1:])), p=1, dim=1)

            mask = filter_norms.gt(threshold).type(module.weight.data.type())
            mask = mask.expand(np.prod(module.weight.shape[1:]), module.weight.shape[0]).t()
            mask = mask.view(*module.weight.shape)
            # apply the pruning to the layer
            module.weight.data.mul_(mask) 
            setattr(module, 'pruning_metadata', {'mask': mask})

    def prune_channels(self, model, pruning_ratio):
        """Prune channels globally based on the given pruning ratio
        """
        norms = torch.tensor([], device=model.device, requires_grad=False)
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
            if module.weight.dim() == 4:
                mask = mask.expand(module.weight.shape[0], module.weight.shape[1]).unsqueeze(-1)
                mask = mask.expand(module.weight.shape[0],
                                   module.weight.shape[1],
                                   module.weight.shape[2] * module.weight.shape[3]).contiguous()
                mask = mask.view(*module.weight.shape)
            elif module.weight.dim() == 2:
                mask = mask.expand(*module.weight.shape)

            # apply the pruning to the layer
            module.weight.data.mul_(mask) 
            setattr(module, 'pruning_metadata', {'mask': mask})

    def prune_kernels(self, model, pruning_ratio):
        """Prune kernels globally based on the given pruning ratio
        """
        # TODO: Consider adding kernel pruning
        raise NotImplementedError

    def prune_weights(self, model, pruning_ratio):
        """Prune weights globally based on the given pruning ratio
        """
        all_weights = torch.tensor([], device=model.device, requires_grad=False)
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
            setattr(module, 'pruning_metadata', {'mask': mask})

    def prune_eridanus_blocks(self, model, pruning_ratio, window_w=None, window_h=None):
        """Prune Eridanus-like blocks globally based on the given pruning ratio. The dimensions
           of the sliding window are given, with the most important being the window width,
           which should theoretically match the width of the systolic array
        """
        def get_weights_2d(module):
            with torch.no_grad():
               return module.weight.view(-1, np.prod(module.weight.shape[1:]))

        if window_w is None:
            assert self.args.eridanus_window_w is not None
            window_w = self.args.eridanus_window_w
        if window_h is None:
            assert self.args.eridanus_window_h is not None
            window_h = self.args.eridanus_window_h

        # get the norm of every possible block in the weight matric of each layer
        all_blocks = torch.tensor([], device=model.device, requires_grad=False)
        for name, module in model.named_modules():
            if name not in self.layers_to_compress:
                continue

            i_w = i_h = 0
            weights_2d = get_weights_2d(module)
            while i_w < weights_2d.shape[0]:
                with torch.no_grad():
                    block = weights_2d[i_w : (i_w + window_w - 1), i_h : (i_h + window_h - 1)]
                    all_blocks = torch.cat((all_blocks, torch.norm(block, p=1).unsqueeze(0)))

                # slide in the height dimension
                i_h += 1
                # slide in the width dimension
                if i_h > weights_2d.shape[1] - window_h:
                    i_w += window_w
                    i_h = 0

        # find the threshold value for all block norms
        threshold = torch.quantile(all_blocks.squeeze(), pruning_ratio)
        logger.debug(f"Pruning ratio {pruning_ratio} turned to threshold {threshold:.3e}")

        for name, module in model.named_modules():
            if name not in self.layers_to_compress:
                continue
            
            # follow the exact Eridanus algorithm here, having established the threshold
            i_w = i_h = 0
            weights_2d = get_weights_2d(module)
            mask = torch.ones_like(weights_2d)
            while i_w < weights_2d.shape[0]:
                block = weights_2d[i_w : (i_w + window_w - 1), i_h : (i_h + window_h - 1)]

                # compare the block norm against the threshold
                if torch.norm(block, p=1) < threshold:
                    mask[i_w : (i_w + window_w - 1), i_h : (i_h + window_h - 1)] = 0
                    i_h += window_h
                else:
                    i_h += 1
                # slide in the width dimension
                if i_h > weights_2d.shape[1] - window_h:
                    i_w += window_w
                    i_h = 0

            # reshape the mask and apply pruning
            mask = mask.view(*module.weight.shape)
            module.weight.data.mul_(mask)
            setattr(module, 'pruning_metadata', {'mask': mask})

    def reset(self):
        pass

