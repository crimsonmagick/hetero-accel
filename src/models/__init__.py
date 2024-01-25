import torch
import logging
import torchvision.models as torch_models
from . import image_classification as ic_models
from . import image_segmentation as is_models
from . import object_detection as od_models
from . import language_processing as lp_models
from . import video_processing as vp_models


logger = logging.getLogger(__name__)


def create_model(arch, dataset, batch_size=256, pretrained=True, parallel=True, device_ids=None, verbose=True):
    """Create a PyTorch model based on the given architecture and dataset
    """
    def get_model_input_shape():
        """Determine input shape based on classification dataset"""
        if 'inception_v3' in arch:
            return (batch_size, 3, 299, 299)
        try:
            return {'imagenet': (batch_size, 3, 224, 224),
                    'cifar10': (batch_size, 3, 32, 32),
                    'cifar100': (batch_size, 3, 32, 32),
                    'mnist': (batch_size, 1, 28, 28),
                    'voc_seg': None,
                    'voc_det': (batch_size, 3, 320, 320),
                    'coco': (),
                    'sst2': (),
                    'multi30k': (),
                    'moving_mnist': (),
                    'kinetics': (),
                   }.get(dataset)
        except KeyError:
            logger.error(f"Input shape for dataset {dataset} could not be determined")
            exit(1)

    def assign_layer_names():
        """Assign human-readable names to the modules (layers)"""
        for name, module in model.named_modules():
            module.full_name = name

    # initialize model from the available architectures per dataset
    is_image_classifier = dataset in ['mnist', 'cifar10', 'cifar100', 'imagenet']
    create_model_f = {'mnist': create_image_classification_model,
                      'cifar10': create_image_classification_model,
                      'cifar100': create_image_classification_model,
                      'imagenet': create_image_classification_model,
                      'voc_seg': create_image_segmentation_model,
                      'voc_det': create_object_detection_model,
                      'coco': create_object_detection_model,
                      'sst2': create_language_processing_model,
                      'multi30k': create_language_processing_model,
                      'moving_mnist': create_video_processing_model,
                      'kinetics': create_video_processing_model,
                     }.get(dataset)
    assert create_model_f is not None, f"Dataset {dataset} is not supported"
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
        #model.is_parallel = parallel
    else:
        device = 'cpu'
        #model.is_parallel = False

    assign_layer_names()
    model.input_shape = get_model_input_shape()
    model.arch = arch
    model.dataset = dataset
    model.device = device
    model.is_image_classifier = is_image_classifier

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


def create_image_segmentation_model(arch, dataset, pretrained):
    return is_models.__dict__[arch](pretrained=pretrained)


def create_language_processing_model(arch, dataset, pretrained):
    return lp_models.__dict__[arch](pretrained=pretrained)


def create_object_detection_model(arch, dataset, pretrained):
    return od_models.__dict__[arch](pretrained=pretrained)


def create_video_processing_model(arch, dataset, pretrained):
    return vp_models.__dict__[arch](pretrained=pretrained)
