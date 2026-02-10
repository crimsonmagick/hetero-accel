import torch
import logging
import torchvision.models as torch_models
from enum import Enum
from . import image_classification as ic_models


logger = logging.getLogger(__name__)


class DNNType(Enum):
    ImageClassification = 1
    SemanticSegmantation = 2
    ObjectDetection = 3
    TextClassification = 4
    MachineTranslation = 5
    VideoProcessing = 6


def create_model(arch, dataset, batch_size=256, pretrained=True, parallel=True, device_ids=None, verbose=True):
    """Create a PyTorch model based on the given architecture and dataset
    """
    def assign_layer_names():
        """Assign human-readable names to the modules (layers)"""
        for name, module in model.named_modules():
            module.full_name = name

    # initialize model from the available architectures per dataset
    # TODO: Determine input shapes per dataset 
    try:
        create_model_f, model_type, input_shape = {
            'mnist': (create_image_classification_model, DNNType.ImageClassification, (batch_size, 1, 28, 28)),
            'cifar10': (create_image_classification_model, DNNType.ImageClassification, (batch_size, 3, 32, 32)),
            'cifar100': (create_image_classification_model, DNNType.ImageClassification, (batch_size, 3, 32, 32)),
            'imagenet': (create_image_classification_model, DNNType.ImageClassification, (batch_size, 3, 224, 224)),
        }[dataset]
    except KeyError:
        raise ValueError(f"Dataset {dataset} is not supported")

    # create the model
    model = create_model_f(arch, dataset, pretrained)

    if verbose:
        logger.info("=> created a {}{} model with the {} dataset".format('pretrained ' if pretrained else '', arch, dataset))

    # configure device and parallel model
    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if parallel:
            if arch.startswith('alexnet'):  # or arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
            else:
                model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device = 'cpu'

    assign_layer_names()
    setattr(model, 'input_shape', input_shape)
    setattr(model, 'arch', arch)
    setattr(model, 'dataset', dataset)
    setattr(model, 'device', device)
    setattr(model, 'task', model_type)
    return model.to(device)


def create_image_classification_model(arch, dataset, pretrained):
    """Create an image classifier
    """
    def _create_cifar_model():
        mobilenet_cifar_models = {**ic_models.mobilenet_cifar_model.__dict__, 
                                  **ic_models.mobilenetv2_cifar_model.__dict__}
        cifar_models = {**ic_models.vgg_cifar_models.__dict__, 
                        **ic_models.resnet_cifar_models.__dict__,
                        **mobilenet_cifar_models, 
                        **ic_models.efficientnet_cifar_model.__dict__,
                        **ic_models.alexnet_cifar_model.__dict__}
        try:
            return cifar_models[arch + '_' + dataset](pretrained=False)
        except KeyError:
            raise NotImplementedError(f'Model {arch} is not supported for the CIFAR dataset')

    def _create_imagenet_model():
        tiny = 'tiny' in dataset
        num_classes = 200 if tiny else 1000
        try:
            weights = None if not pretrained else 'DEFAULT'
            model = torch_models.__dict__[arch](weights=weights, num_classes=num_classes)
        except KeyError:
            raise NotImplementedError('Model {} is not supported for the {}Imagenet dataset'.format(arch, 'Tiny-' if tiny else ''))
        return model

    def _create_mnist_model():
        return ic_models.mnist_models.__dict__[arch + '_' + 'mnist']()

    # load the correct function
    if 'cifar' in dataset:
        return _create_cifar_model()
    elif 'imagenet' in dataset:
        return _create_imagenet_model()
    else:
        return _create_mnist_model()