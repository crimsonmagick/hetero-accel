from torchvision.models import segmentation as seg


def fcn_resnet50(pretrained=True):
    """Semantic segmentation with FCN ResNet50
    """
    weights = seg.FCN_ResNet50_Weights.DEFAULT if pretrained else None
    model = seg.fcn_resnet50(weights=weights, progress=False)
    return model


def fcn_resnet101(pretrained=True):
    """Semantic segmentation with FCN ResNet5101
    """
    weights = seg.FCN_ResNet101_Weights.DEFAULT if pretrained else None
    model = seg.fcn_resnet101(weights=weights, progress=False)
    return model
