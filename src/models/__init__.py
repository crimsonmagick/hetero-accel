import torch
import logging
import torchvision.models as torch_models
from . import lenet_mnist as lenet_mnist_model
from . import alexnet_cifar as alexnet_cifar_model
from . import vgg_cifar_playground as vgg_cifar_models
from . import resnet_cifar_playground as resnet_cifar_models
from . import mobilenet_cifar_playground as mobilenet_cifar_model
from . import mobilenetv2_cifar_playground as mobilenetv2_cifar_model
from . import efficientnet_cifar_playground as efficientnet_cifar_model

logger = logging.getLogger(__name__)

TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(torch_models.__dict__[name]))
VGG_CIFAR_MODEL_NAMES = sorted(name for name in vgg_cifar_models.__dict__
                               if name.islower() and 'vgg' in name and not name.startswith('__')
                               and callable(vgg_cifar_models.__dict__[name]))
RESNET_CIFAR_MODEL_NAMES = sorted(name for name in resnet_cifar_models.__dict__ 
                                  if name.islower() and 'resnet' in name and not name.startswith('__')
                                  and callable(resnet_cifar_models.__dict__[name]))
ALEXNET_CIFAR_MODEL_NAME = sorted(name for name in alexnet_cifar_model.__dict__
                                  if name.islower() and 'alexnet' in name and not name.startswith('__')
                                  and callable(alexnet_cifar_model.__dict__[name]))
mobilenet_cifar_models = {**mobilenet_cifar_model.__dict__, **mobilenetv2_cifar_model.__dict__}
MOBILENET_CIFAR_MODEL_NAMES = sorted(name for name in mobilenet_cifar_models
                                     if name.islower() and 'mobilenet' in name and not name.startswith('__')
                                     and callable(mobilenet_cifar_models[name]))
EFFICIENTNET_CIFAR_MODEL_NAME = sorted(name for name in efficientnet_cifar_model.__dict__
                                    if name.islower() and 'efficientnet' in name and not name.startswith('__')
                                    and callable(efficientnet_cifar_model.__dict__[name]))
LENET_MNIST_MODEL_NAME = sorted(name for name in lenet_mnist_model.__dict__
                                if name.islower() and 'lenet' in name and not name.startswith('__')
                                and callable(lenet_mnist_model.__dict__[name]))

ALL_MODELS = sorted(map(lambda s: s.lower(),
                        set(
                            TORCHVISION_MODEL_NAMES +
                            VGG_CIFAR_MODEL_NAMES + RESNET_CIFAR_MODEL_NAMES +
                            MOBILENET_CIFAR_MODEL_NAMES + EFFICIENTNET_CIFAR_MODEL_NAME +
                            ALEXNET_CIFAR_MODEL_NAME +
                            LENET_MNIST_MODEL_NAME)))


def create_model(arch, dataset, pretrained=True, parallel=True, device_ids=None, verbose=True):
    """Create a PyTorch model based on the given architecture and dataset
    """
    def get_model_input_shape():
        """Determine input shape based on classification dataset"""
        if arch == 'inception_v3':
            return (1, 3, 299, 299)
        if dataset.lower() == 'imagenet':
            return (1, 3, 224, 224)
        elif 'cifar' in dataset.lower():
            return (1, 3, 32, 32)
        elif dataset.lower() == 'mnist':
            return (1, 1, 28, 28)
        else:
            raise ValueError(f"Dataset {dataset} is not supported")

    def assign_layer_names():
        """Assign human-readable names to the modules (layers)"""
        for name, module in model.named_modules():
            module.full_name = name

    # initialize model from the available architectures per dataset
    if 'cifar' in dataset.lower():
        model = _create_cifar_model(arch, pretrained=False,
                                    num_classes=int(dataset.lower().replace('cifar', '')))
    elif dataset.lower() in ['tiny-imagenet', 'imagenet']:
        model = _create_imagenet_model(arch, pretrained, tiny='tiny' in dataset.lower())
    elif dataset.lower() == 'mnist':
        model = _create_mnist_model(arch, pretrained)

    if verbose:
        logger.info(
            "=> created a {}{} model with the {} dataset".format('pretrained ' if pretrained else '', arch, dataset)
        )

    # configure device and parallel model
    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if parallel:
            if arch.startswith('alexnet'):  # or arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
            else:
                model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.is_parallel = parallel
    else:
        device = 'cpu'
        model.is_parallel = False

    assign_layer_names()
    model.input_shape = get_model_input_shape()
    model.arch = arch
    model.dataset = dataset
    model.device = device

    return model.to(device)


def _create_cifar_model(arch, pretrained, num_classes):
    cifar_models = {**vgg_cifar_models.__dict__,  **resnet_cifar_models.__dict__,
                    **mobilenet_cifar_models, **efficientnet_cifar_model.__dict__,
                    **alexnet_cifar_model.__dict__}
    try:
        model = cifar_models[arch + '_cifar'](pretrained=pretrained, num_classes=num_classes)
    except KeyError:
        raise NotImplementedError(f'Model {arch} is not supported for the CIFAR{num_classes} dataset')
    return model


def _create_imagenet_model(arch, pretrained, tiny=False):
    num_classes = 200 if tiny else 1000
    if arch in TORCHVISION_MODEL_NAMES:
        weights = None if not pretrained else 'DEFAULT'
        model = torch_models.__dict__[arch](weights=weights, num_classes=num_classes)
    else:
        raise NotImplementedError('Model {} is not supported for the {}Imagenet dataset'.format(arch, 'Tiny-' if tiny else ''))
    return model


def _create_mnist_model(arch, pretrained):
    model = lenet_mnist_model.__dict__[arch](pretrained)
    return model

