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


def create_model(arch, dataset, batch_size=256, pretrained=True, parallel=True, device_ids=None, verbose=True):
    """Create a PyTorch model based on the given architecture and dataset
    """
    def get_model_input_shape():
        """Determine input shape based on classification dataset"""
        if 'inception_v3' in arch:
            return (batch_size, 3, 299, 299)
        if dataset == 'imagenet':
            return (batch_size, 3, 224, 224)
        elif 'cifar' in dataset:
            return (batch_size, 3, 32, 32)
        elif dataset == 'mnist':
            return (batch_size, 1, 28, 28)

    def assign_layer_names():
        """Assign human-readable names to the modules (layers)"""
        for name, module in model.named_modules():
            module.full_name = name

    # initialize model from the available architectures per dataset
    is_image_classifier = True
    if 'cifar' in dataset:
        model = _create_cifar_model(arch, pretrained)
    elif dataset in ['tiny-imagenet', 'imagenet']:
        arch = arch.replace('tiny', '').replace('_imagenet', '')
        model = _create_imagenet_model(arch, pretrained, tiny='tiny' in dataset)
    elif dataset == 'mnist':
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
    model.is_image_classifier = is_image_classifier

    return model.to(device)


def _create_cifar_model(arch, pretrained):
    mobilenet_cifar_models = {**mobilenet_cifar_model.__dict__, 
                              **mobilenetv2_cifar_model.__dict__}
    cifar_models = {**vgg_cifar_models.__dict__,  **resnet_cifar_models.__dict__,
                    **mobilenet_cifar_models, **efficientnet_cifar_model.__dict__,
                    **alexnet_cifar_model.__dict__}
    try:
        return cifar_models[arch](pretrained=False)
    except KeyError:
        raise NotImplementedError(f'Model {arch} is not supported for the CIFAR dataset')


def _create_imagenet_model(arch, pretrained, tiny=False):
    num_classes = 200 if tiny else 1000
    try:
        weights = None if not pretrained else 'DEFAULT'
        model = torch_models.__dict__[arch](weights=weights, num_classes=num_classes)
    except KeyError:
        raise NotImplementedError('Model {} is not supported for the {}Imagenet dataset'.format(arch, 'Tiny-' if tiny else ''))
    return model


def _create_mnist_model(arch, pretrained):
    model = lenet_mnist_model.__dict__[arch](pretrained)
    return model

