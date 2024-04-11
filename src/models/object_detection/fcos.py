from torchvision.models import detection as det


def fcos_resnet50_fpn(pretrained=True):
    """Object detection with a FCOS model with a ResNet-50-FPN backbone.
    """
    weights = det.FCOS_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    return det.fcos_resnet50_fpn(weights=weights)
