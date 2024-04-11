from torchvision.models import detection as det


def ssd300_vgg16(pretrained=True):
    """Object detection with the SSD300 with VGG16 backbone
    """
    weights = det.SSD300_VGG16_Weights.DEFAULT if pretrained else None
    return det.ssd300_vgg16(weights=weights)


def ssdlite320_mobilenet_v3_large(pretrained=True):
    """Object detection with SSDlite model architecture with input size
       320x320 and a MobileNetV3 Large backbone
    """
    weights = det.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    return det.ssdlite320_mobilenet_v3_large(weights=weights)
