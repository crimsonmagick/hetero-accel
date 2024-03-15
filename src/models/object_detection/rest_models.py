from torchvision.models import detection as det


def ssd300_vgg16(pretrained=True):
    """Object detection with the SSD300 with VGG16 backbone
    """
    weights = det.SSD300_VGG16_Weights.DEFAULT if pretrained else None
    return det.ssd300_vgg16(weights=weights)


def retinanet_resnet50(pretrained=True):
    """Object detection with a RetinaNet model with a ResNet-50-FPN backbone
    """
    weights = det.SSD300_VGG16_Weights.DEFAULT if pretrained else None
    return det.retinanet_resnet50_fpn(weights=weights)


def fasterrcnn_resnet50(pretrained=True):
    """Object detection with a Faster R-CNN model with a ResNet-50-FPN backbone
    """
    weights = det.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    return det.fasterrcnn_resnet50_fpn(weights=weights)
