import logging
import torch
from functools import partial
from collections import namedtuple, OrderedDict
from src.utils import transform_model
from src.compression.quantization import linear_quantize, linear_dequantize, get_quant_params


logger = logging.getLogger(__name__)

QuantMetadata = namedtuple('QuantMetadata', ['bits', 'scale', 'zero_point', 'min_q_val', 'max_q_val'])


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

        def replace_conv2d(module, full_name):
            """Replacement function for Conv2d modules"""
            assert isinstance(module, torch.nn.Conv2d)
            new_module = QuantConv2d(in_channels=module.in_channels,
                                     out_channels=module.out_channels,
                                     kernel_size=module.kernel_size,
                                     stride=module.stride,
                                     padding=module.padding,
                                     dilation=module.dilation,
                                     groups=module.groups,
                                     bias=getattr(module, 'bias', None) is not None,
                                     weight_bits=w_bits,
                                     activation_bits=a_bits,
                                     accumulator_bits=32,
                                     signed=self.signed,
                                     symmetric=self.symmetric)
            new_module.weight = module.weight
            new_module.bias = module.bias
            return new_module

        replacement_by_type = OrderedDict()
        replacement_by_type[torch.nn.Conv2d] = replace_conv2d
        replacement_by_name = {name: replacement_by_type[type(module)]
                               for name, module in model.named_modules()
                               if name in self.layers_to_compress and type(module) in replacement_by_type}

        transform_model(model, replacement_by_name, replace_by_name=True)


class QuantConv2d(torch.nn.Conv2d):
    """Wrapper for symmetric, signed, linear quantization of Conv2d modules
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                 weight_bits=8, activation_bits=8, accumulator_bits=32, signed=True, symmetric=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight_bits = weight_bits
        self.act_bits = activation_bits
        self.accum_bits = accumulator_bits
        self.signed = signed
        self.symmetric = symmetric
        self.deactivated = False
        self.weights_quant_metadata = None

        # Quantize weights - overwrite FP32 weights
        with torch.no_grad():
            self.quantize_weights()

    def quantize_weights(self):
        """Quantize the weights
        """
        scale, zero_point, min_q_val, max_q_val = get_quant_params(self.weight, self.weight_bits)
        linear_quantize(self.weight, scale, zero_point, min_q_val, max_q_val, inplace=True)
        self.weights_quant_metadata = QuantMetadata(self.weight_bits, scale, zero_point, min_q_val, max_q_val)

    def forward(self, *inputs):
        if self.training:
            raise RuntimeError(self.__class__.__name__ + " can only be used in eval mode")

        if self.deactivated:
            out = super().forward(*inputs)
            return out

        # prepare inputs for quantization
        inputs_q = [self.quantize_inputs(input) for input in inputs]

        accum = self.quantized_forward(*inputs_q)

         # Re-quantize accumulator to quantized output range
        output_q = self.requantize_accumulator(accum)

        output_fp = self.dequantize_output(output_q)

        return output_fp

    def quantize_inputs(self, input):
        """Quantize the inputs
        """
        if not hasattr(input, 'quant_metadata'):
            scale, zero_point, min_q_val, max_q_val = get_quant_params(input, self.act_bits)
            setattr(input, 'quant_metadata', QuantMetadata(self.act_bits, scale, zero_point, min_q_val, max_q_val))

            q_input = linear_quantize(input,
                                    input.quant_metadata.scale,
                                    input.quant_metadata.zero_point,
                                    input.quant_metadata.min_q_val,
                                    input.quant_metadata.max_q_val,
                                    inplace=False)
        else:
            q_input = input

        # copy metadata to quantized inputs
        setattr(q_input, 'quant_metadata', input.quant_metadata)
        return q_input

    def quantized_forward(self, input_q):
        """Execute the forward pass with rescaled input and quantized weights
        """
        self.accum_scale = input_q.quant_metadata.scale * self.weights_quant_metadata.scale
        input_q += input_q.quant_metadata.zero_point
        accum = super().forward(input_q)
        _, _, min_q_val, max_q_val = get_quant_params(accum, self.accum_bits)
        accum.clamp_(min_q_val, max_q_val)
        return accum

    def requantize_accumulator(self, accum):
        """Re-quantize accumulator to quantized output range
        """
        out_scale, out_zero_point, min_q_val, max_q_val, = get_quant_params(accum / self.accum_scale, self.act_bits)

        # https://github.com/kompalas/NN_project/blob/main/distiller/distiller/quantization/range_linear.py#L972
        requant_scale = out_scale / self.accum_scale

        output_q = linear_quantize(accum, requant_scale, out_zero_point, min_q_val, max_q_val, inplace=True)
        setattr(output_q, 'quant_metadata', QuantMetadata(self.act_bits,
                                                          out_scale,
                                                          out_zero_point,
                                                          min_q_val,
                                                          max_q_val))
        return output_q

    def dequantize_output(self, output_q):
        """Revert output activations back to floating point
        """
        output_fp = linear_dequantize(output_q, output_q.quant_metadata.scale,
                                      output_q.quant_metadata.zero_point, inplace=True)
        setattr(output_fp, 'quant_metadata', output_q.quant_metadata)
        return output_fp
