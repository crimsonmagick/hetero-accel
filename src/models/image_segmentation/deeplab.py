from torchvision.models import segmentation as seg


def deeplabv3_mobilenet_v3_large(pretrained=True):
    """Semantic segmentation with a DeepLabV3 model with a MobileNetV3-Large backbone.
    """
    weights = seg.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = seg.deeplabv3_mobilenet_v3_large(weights=weights, progress=False)
    return model


def deeplabv3_resnet50(pretrained=True):
    """Semantic segmentation with a DeepLabV3 model with a ResNet-50 backbone.
    """
    weights = seg.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = seg.deeplabv3_resnet50(weights=weights, progress=False)
    return model


def deeplabv3_resnet101(pretrained=True):
    """Semantic segmentation with a DeepLabV3 model with a ResNet-101 backbone.
    """
    weights = seg.DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
    model = seg.deeplabv3_resnet101(weights=weights, progress=False)
    return model
