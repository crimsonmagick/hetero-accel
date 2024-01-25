from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

def ssd(pretrained=True):
    weights = SSD300_VGG16_Weights.DEFAULT if pretrained else None
    return ssd300_vgg16(weights=weights)
