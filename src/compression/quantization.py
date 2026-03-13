import logging
import torch
from collections import namedtuple
from argparse import Namespace
import os
import random
import sys
from typing import List
from typing import Optional
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from brevitas.export import export_onnx_qcdq
from brevitas.export import export_torch_qcdq
from brevitas.export.inference import quant_inference_mode
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize
from brevitas_examples.common.parse_utils import override_defaults
from brevitas_examples.common.parse_utils import parse_args
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import apply_learned_round
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_act_equalization
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_bias_correction
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_gpfq
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_gptq
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_qronos
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate_bn
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
from brevitas_examples.imagenet_classification.ptq.ptq_imagenet_args import create_args_parser
from brevitas_examples.imagenet_classification.ptq.ptq_imagenet_args import \
    validate as validate_args
from brevitas_examples.imagenet_classification.ptq.utils import get_model_config
from brevitas_examples.imagenet_classification.utils import SEED
from brevitas_examples.imagenet_classification.utils import validate
from crimson_magick.cifar_zoo import get_test_loader, Cifar, Arch, load_model
from crimson_magick.cifar_zoo.fine_tuned.fine_tuned_models import ArchType
from torchvision import datasets, transforms

GRAPH_EQ_ITERATIONS = 20

logger = logging.getLogger(__name__)

QuantMetadata = namedtuple('QuantMetadata', ['bits', 'scale', 'zero_point', 'min_q_val', 'max_q_val'])


class Quantizer:

    # def __init__(self, test_loader):
    #     self.test_loader = test_loader

    def reset(self):
        pass

    def quantize(self, model, q_bits):
        q_bits = int(q_bits)
        dtype = next(model.parameters()).dtype
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Get model-specific configurations about input shapes and normalization
        input_shape = 224
        resize_shape = 256

        calib_loader = generate_cifar100_calib_loader(
            'data'
        )

        cudnn.benchmark = False
        model.eval()

        # Preprocess the model for quantization
        target_backend = 'layerwise' if hasattr(model, 'arch_type') and model.arch_type == ArchType.MOBILENET else 'fx'
        model = preprocess_for_quantize(
            model,
            equalize_iters=GRAPH_EQ_ITERATIONS,
            equalize_merge_bias=True,
            merge_bn=True,
            channel_splitting_ratio=0.0,
            channel_splitting_split_input=False)

        device = next(iter(model.parameters())).device

        act_quant_percentile = 99.999

        # Define the quantized model
        quant_model = quantize_model(
            model,
            dtype=dtype,
            device=device,
            backend=target_backend,
            scale_factor_type='float_scale',
            bias_bit_width=32, # TODO should this match other parameters?
            weight_bit_width=q_bits,
            weight_narrow_range=False,
            weight_param_method='stats',
            weight_quant_granularity='per_tensor',
            act_quant_granularity='per_tensor',
            weight_quant_type='sym',
            layerwise_first_last_bit_width=8,
            act_bit_width=q_bits,
            act_param_method='stats',
            act_quant_percentile=act_quant_percentile,
            act_quant_type='sym',
            quant_format='int',
            layerwise_first_last_mantissa_bit_width=4,
            layerwise_first_last_exponent_bit_width=3,
            weight_mantissa_bit_width=4,
            weight_exponent_bit_width=3,
            act_mantissa_bit_width=4,
            act_exponent_bit_width=3,
            act_scale_computation_type='static',
            uint_sym_act_for_unsigned_values=True)

        # Some quantizer configurations require a forward pass to initialize scale factors.
        # This forward pass ensures that subsequent algorithms work as intended
        model.eval()
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        images, _ = next(iter(calib_loader))
        images = images.to(device=device, dtype=dtype)
        with torch.no_grad():
            model(images)

        # Calibrate the quant_model on the calibration dataloader
        print("Starting activation calibration:")
        calibrate(calib_loader, quant_model)

        # if args.gpfq:
        #     print("Performing GPFQ:")
        #     apply_gpfq(
        #         calib_loader,
        #         quant_model,
        #         act_order=args.gpxq_act_order,
        #         max_accumulator_bit_width=args.gpxq_accumulator_bit_width,
        #         max_accumulator_tile_size=args.gpxq_accumulator_tile_size)
        #
        # if args.gptq:
        #     print("Performing GPTQ:")
        #     apply_gptq(
        #         calib_loader,
        #         quant_model,
        #         act_order=args.gpxq_act_order,
        #         create_weight_orig=not args.disable_create_weight_orig,
        #         use_quant_activations=args.gptq_use_quant_activations,
        #         max_accumulator_bit_width=args.gpxq_accumulator_bit_width,
        #         max_accumulator_tile_size=args.gpxq_accumulator_tile_size)

        # print("Calibrate BN:")
        # calibrate_bn(calib_loader, quant_model)

        print("Applying bias correction:")
        apply_bias_correction(calib_loader, quant_model)

        # Validate the quant_model on the validation dataloader
        # print("Starting validation:")
        with torch.no_grad(), quant_inference_mode(quant_model):
            param = next(iter(quant_model.parameters()))
            device, dtype = param.device, param.dtype
            ref_input = generate_ref_input(device, dtype, input_shape)
            quant_model(ref_input)
            compiled_model = torch.compile(quant_model, fullgraph=True, disable=True)
            test_loader = get_test_loader(Cifar.CIFAR100)
            quant_top1 = validate(test_loader, compiled_model, stable=dtype != torch.bfloat16)
            print(f"IN QUANTIZER - quant_top1={quant_top1}")
            return compiled_model

        # return {"quant_top1": float(quant_top1)}, quant_model


# Ignore warnings about __torch_function__
warnings.filterwarnings("ignore")

def generate_ref_input(device, dtype, input_shape):
    return torch.ones(1, 3, input_shape, input_shape, device=device, dtype=dtype)


def generate_cifar100_calib_loader(
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 8,
        n_calib: int = 2048
):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=tfm)
    subset = torch.utils.data.Subset(ds, list(range(min(n_calib, len(ds)))))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return loader


def generate_cifar10_calib_loader(
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 8,
        n_calib: int = 2048
):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
    subset = torch.utils.data.Subset(ds, list(range(min(n_calib, len(ds)))))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return loader