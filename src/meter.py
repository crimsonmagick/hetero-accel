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
    metrics:
        Pixel accuaracy 
        mIOU: mean IOU for classes 
    """
    def __init__(self, num_classes = 21):
        self.metrics = ['pixel_acc', 'mIOU']
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.pixel_acc_sum = 0
        self.conf_mat = None
        self.num = 0

    def add(self, output, target):
        acc = self.pixel_acc(output, target)
        self.pixel_acc_sum += acc
        self.iou(output, target)
        self.num += 1

    def value(self, metric=None):
        assert metric in self.metrics + ['all'], f'Metric {metric} is not supported'
        
        # get iou for every class  
        iu = torch.diag(self.conf_mat) / (self.conf_mat.sum(1) + self.conf_mat.sum(0) - torch.diag(self.conf_mat))

        if metric == 'all':
            return self.pixel_acc_sum / self.num, iu.mean()
        elif metric == 'pixel_acc':
            return self.pixel_acc_sum / self.num,

        elif metric == 'mIOU':
            return iu.mean(),
    
    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = ((label >= 0) & (label < self.num_classes)).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc
    
    def iou(self, pred, label):
        _, pred = torch.max(pred, dim=1)
        n = self.num_classes
        if self.conf_mat is None:
            self.conf_mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.inference_mode():
            pred, label = pred.flatten(), label.flatten()
            k = (label >= 0) & (label < n)
            inds = n * label[k].to(torch.int64) + pred[k]
            self.conf_mat += torch.bincount(inds, minlength=n**2).reshape(n, n)


class ObjectDetectionMeter(Meter):
    """Accuracy meter for object detection
    """
    def __init__(self):
        raise NotImplementedError
