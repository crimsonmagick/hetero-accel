import logging
import random
import warnings
from collections import namedtuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from brevitas.export.inference import quant_inference_mode
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_bias_correction
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_gptq
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
from brevitas_examples.imagenet_classification.utils import SEED
from crimson_magick.cifar_zoo.fine_tuned.fine_tuned_models import ArchType
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100

from src.datasets.imagenet_dataset import ImagenetDataset

GRAPH_EQ_ITERATIONS = 20

logger = logging.getLogger(__name__)

QuantMetadata = namedtuple('QuantMetadata', ['bits', 'scale', 'zero_point', 'min_q_val', 'max_q_val'])


class Quantizer:

    # def __init__(self, test_loader):
    #     self.test_loader = test_loader
    def get_quant_config(self, model, dataset):
        config = {}
        if type(dataset) is ImagenetDataset:
            config['percentile'] = 99.95
            config['target_backend'] = 'layerwise'
            config['merge_bn'] = True
        elif hasattr(model, 'arch_type') and model.arch_type == ArchType.MOBILENET:
            config['percentile'] = 99.95
            config['gptq'] = True
            config['target_backend'] = 'layerwise'
            config['merge_bn'] = False
        elif type(dataset) is CIFAR100:
            config['percentile'] = 99.95
            config['gptq'] = True
            config['target_backend'] = 'fx'
            config['merge_bn'] = True
        else:
            config['percentile'] = 99.999
            config['gptq'] = True
            config['target_backend'] = 'fx'
            config['merge_bn'] = True

        if hasattr(model, "arch") and 'inception' in model.arch.lower():
            config['size'] = 299
        else:
            config['size'] = 224

        return config

    def reset(self):
        pass

    def quantize(self, model, q_bits, dataset):
        q_bits = int(q_bits)
        dtype = next(model.parameters()).dtype
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Get model-specific configurations about input shapes and normalization

        calib_loader = generate_calib_loader(dataset)
        quant_config = self.get_quant_config(model, dataset)

        cudnn.benchmark = False
        model.eval()

        # Preprocess the model for quantization
        model = preprocess_for_quantize(
            model,
            trace_model=True,
            equalize_iters=GRAPH_EQ_ITERATIONS,
            equalize_merge_bias=True,
            merge_bn=quant_config['merge_bn'],
            channel_splitting_ratio=0.0,
            channel_splitting_split_input=False)

        device = next(iter(model.parameters())).device

        act_quant_percentile = quant_config['percentile']

        # Define the quantized model
        quant_model = quantize_model(
            model,
            dtype=dtype,
            device=device,
            backend=quant_config['target_backend'],
            scale_factor_type='float_scale',
            bias_bit_width=32,  # TODO should this match other parameters?
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

        if quant_config.get('gptq'):
            print("Performing GPTQ:")
            apply_gptq(
                calib_loader,
                quant_model,
                act_order=False,
                create_weight_orig=True,
                use_quant_activations=False,
                max_accumulator_bit_width=None,
                max_accumulator_tile_size=None)

        print("Applying bias correction:")
        apply_bias_correction(calib_loader, quant_model)

        # Validate the quant_model on the validation dataloader
        # print("Starting validation:")
        with torch.no_grad(), quant_inference_mode(quant_model):
            param = next(iter(quant_model.parameters()))
            device, dtype = param.device, param.dtype
            ref_input = generate_ref_input(device, dtype, quant_config['size'])
            quant_model(ref_input)
            compiled_model = torch.compile(quant_model, fullgraph=True, disable=True)
            del calib_loader
            return compiled_model


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


def generate_calib_loader(
        dataset,
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

    subset = torch.utils.data.Subset(dataset, list(range(min(n_calib, len(dataset)))))
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
