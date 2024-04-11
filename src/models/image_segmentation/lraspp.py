from torchvision.models import segmentation as seg


def lraspp_mobilenet_v3_large(pretrained=True):
    """Semantic segmentation with a Lite R-ASPP Network model with a MobileNetV3-Large backbone
    """
    weights = seg.LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = seg.lraspp_mobilenet_v3_large(weights=weights, progress=False)
    return model
