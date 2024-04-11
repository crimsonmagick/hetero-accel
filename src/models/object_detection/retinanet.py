from torchvision.models import detection as det


def retinanet_resnet50_fpn(pretrained=True):
    """Object detection with a RetinaNet model with a ResNet-50-FPN backbone.
    """
    weights = det.RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    return det.retinanet_resnet50_fpn(weights=weights)


def retinanet_resnet50_fpn_v2(pretrained=True):
    """Object detection with an improved RetinaNet model with a ResNet-50-FPN backbone.
    """
    weights = det.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    return det.retinanet_resnet50_fpn_v2(weights=weights)
