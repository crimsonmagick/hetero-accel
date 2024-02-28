import torch
from torchnet.meter.meter import Meter
from torchnet.meter import ClassErrorMeter
from src.dataset_utils import VOC_CLASSES_MAP, REV_VOC_CLASSES_MAP


# NOTE: Write first the most important accuracy metric, and the
#       one returned FIRST by calling meter.value('all'). This is
#       important when either train/validate are called from
#       src/train_test.py.

# NOTE: Remember that the value() method should always return a tuple


class ImageClassificationMeter(ClassErrorMeter):
    def __init__(self):
        self.metrics = ['top1', 'top5']
        super().__init__(topk=[1, 5], accuracy=True)

    def value(self, metric=None):
        assert metric in self.metrics + ['all']
        if metric == 'all':
            return tuple([super(ImageClassificationMeter, self).value(k=_k) for _k in [1, 5]])
        return super(ImageClassificationMeter, self).value(k=int(metric.replace('top', '')))


class SegmentationMeter(Meter):
    """Accuracy meter for semantic/image segmantation
    """
    def __init__(self):
        self.metrics = ['iou', 'pixel_acc']
        self.reset()

    def reset(self):
        self.pixel_acc_sum = 0
        self.num = 0

    def add(self, output, target):
        acc = self.pixel_acc(output, target)
        self.pixel_acc_sum += acc
        self.num += 1

    def value(self, metric=None):
        assert metric in self.metrics + ['all'], f'Metric {metric} is not supported'
        if metric == 'all':
            return self.pixel_acc_sum / self.num,
        elif metric == 'pixel_acc':
            return self.pixel_acc_sum / self.num,
    
    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class ObjectDetectionMeter(Meter):
    """Accuracy meter for object detection
    """
    def __init__(self):
        raise NotImplementedError
