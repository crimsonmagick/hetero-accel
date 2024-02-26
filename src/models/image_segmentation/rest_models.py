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


def deeplabv3(pretrained=True):
    """Semantic segmentation with DeepLab V3
    """
    weights = seg.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = seg.deeplabv3_mobilenet_v3_large(weights=weights, progress=False)
    return model
