import logging
import torch
from functools import partial
from collections import namedtuple, OrderedDict
from src.utils import transform_model


logger = logging.getLogger(__name__)

QuantMetadata = namedtuple('QuantMetadata', ['scale', 'zero_point', 'min_q_val', 'max_q_val'])


class Quantizer:
    def __init__(self, layers_to_compress, signed_quant=True, symmetric_quant=True):
        self.layers_to_compress = layers_to_compress
        self.signed = signed_quant
        self.symmetric = symmetric_quant
        self.hook_handles = []

    def reset(self):
        for handle in self.hook_handles:
            handle.remove()

    def quantize(self, model, quant_bits):
        """Apply quantization to the entire forward pass. We use the same 
           number of quantization bits for weights and activations
        """
        # TODO: Consider switching to 8 bits constant for activations
        w_bits = a_bits = quant_bits

        for name, module in model.named_modules():
            if name not in self.layers_to_compress:
                continue

            with torch.no_grad():
                self.quantize_inputs(module, a_bits)
                self.quantize_weights(module, w_bits)
                self.dequantize_outputs(module, a_bits)

    def quantize_inputs(self, module, quant_bits):
        """Quantize inputs of a given layer
        """
        # register the hook during the forward pass
        hook_fn = partial(input_quant_hook, quant_bits=quant_bits)
        handle = module.register_forward_pre_hook(hook_fn)
        self.hook_handles.append(handle)

    def quantize_weights(self, module, quant_bits):
        """Convert floating point weights of a given module to integer
        """
        # hook to rescale inputs before forward pass
        handle = module.register_forward_pre_hook(input_rescale_hook)
        self.hook_handles.append(handle)

        # quantize weights and get ready for the forward pass
        scale, zero_point, min_q_val, max_q_val = get_quant_params(module.weight, quant_bits)
        setattr(module, 'quant_metadata', QuantMetadata(scale, zero_point, min_q_val, max_q_val))
        linear_quantize(module.weight, scale, zero_point, min_q_val, max_q_val, inplace=True)

    def dequantize_outputs(self, module, quant_bits):
        """Revert output activations of a given layer back to floating point
        """
        # hook to clamp the output of the forward function
        hook_fn = partial(quant_dequant_output_hook, quant_bits=quant_bits)
        handle = module.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)


def input_quant_hook(module, input, quant_bits):
    # TODO: this is a workaround, check how to take q_inputs as an iterable of tensors
    assert isinstance(input, (list, tuple)) and len(input) == 1
    input = input[0]
    scale, zero_point, min_q_val, max_q_val = get_quant_params(input, quant_bits)
    q_input = linear_quantize(input, scale, zero_point, min_q_val, max_q_val, inplace=False)
    setattr(q_input, 'quant_metadata', QuantMetadata(scale, zero_point, min_q_val, max_q_val))
    return q_input


def input_rescale_hook(module, q_input):
    # TODO: this is a workaround, check how to take q_inputs as an iterable of tensors
    assert isinstance(q_input, (list, tuple)) and len(q_input) == 1
    q_input = q_input[0]

    assert(hasattr(q_input, 'quant_metadata') and hasattr(module, 'quant_metadata'))
    q_input.accum_scale = q_input.quant_metadata.scale * module.quant_metadata.scale
    q_input += q_input.quant_metadata.zero_point
    return q_input
        

def quant_dequant_output_hook(module, q_input, accum, quant_bits):
    # TODO: this is a workaround, check how to take q_inputs as an iterable of tensors
    assert isinstance(q_input, (list, tuple)) and len(q_input) == 1
    q_input = q_input[0]

    _, _, min_q_val, max_q_val = get_quant_params(accum, 32)
    accum.clamp_(min_q_val, max_q_val)
    out_scale, out_zero_point, min_q_val, max_q_val, = get_quant_params(accum/q_input.accum_scale, quant_bits)
    # https://github.com/kompalas/NN_project/blob/main/distiller/distiller/quantization/range_linear.py#L972
    requant_scale = out_scale / q_input.accum_scale
    q_output = linear_quantize(accum, requant_scale, out_zero_point, min_q_val, max_q_val, inplace=True)
    fp_output = linear_dequantize(q_output, out_scale, out_zero_point, inplace=True)
    setattr(fp_output, 'quant_metadata', QuantMetadata(out_scale, out_zero_point, min_q_val, max_q_val))
    return fp_output



### Utility quantization functions ###

def get_quant_params(param, quant_bits):
    """Get the scale and zero point from a given parameter
    """
    # TODO: The following is only for symmetric signed quantization
    max_q_val = 2**(quant_bits - 1) - 1 
    min_q_val = (-max_q_val) - 1
    # https://github.com/kompalas/NN_project/blob/main/distiller/distiller/quantization/q_utils.py#L141
    sat_val = torch.max(param.min().abs(), param.max().abs())
    # https://github.com/kompalas/NN_project/blob/main/distiller/distiller/quantization/q_utils.py#L46
    n = (2 ** quant_bits - 1) // 2
    try:
        scale = n/sat_val
    except ZeroDivisionError:
        scale = 1
    # zero point is always set to zero
    zero_point = torch.zeros_like(scale)
    return scale, zero_point, min_q_val, max_q_val


def linear_quantize(param, scale, zero_point, min_q_val, max_q_val, inplace=True):
    """Apply linear quantization on a given parameter
    """
    if inplace:
        param.mul_(scale).sub_(zero_point).round_().clamp_(min_q_val, max_q_val)
        return param
    q_param = torch.round(param * scale - zero_point)
    return torch.clamp(q_param, min_q_val, max_q_val)


def linear_dequantize(param, scale, zero_point, inplace=True):
    """Dequantize a given parameter using the scale and zero point
    """
    if inplace:
        param.add_(zero_point).div_(scale)
        return param
    return (param + zero_point) / scale

