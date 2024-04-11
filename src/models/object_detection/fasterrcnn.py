from torchvision.models import detection as det

def fasterrcnn_resnet50_fpn(pretrained=True):
    """Object detection with a Faster R-CNN model with a ResNet-50-FPN backbone
    """
    weights = det.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    return det.fasterrcnn_resnet50_fpn(weights=weights)


def fasterrcnn_resnet50_fpn_v2(pretrained=True):
    """Object detection with an improved Faster R-CNN model with a ResNet-50-FPN backbone
    """
    weights = det.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    return det.fasterrcnn_resnet50_fpn_v2(weights=weights)


def fasterrcnn_mobilenet_v3_large_fpn(pretrained=True):
    """Object detection with a Faster R-CNN model with a MobileNet-V3-FPN backbone
    """
    weights = det.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    return det.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)


def fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True):
    """Object detection with a Faster R-CNN model with a MobileNetV3-Large
       backbone tuned for mobile use cases
    """
    weights = det.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
    return det.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
