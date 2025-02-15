import torch
import copy
import io
from contextlib import redirect_stdout
import numpy as np
from torchnet.meter.meter import Meter
from torchnet.meter import ClassErrorMeter
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.dataset_utils import VOC_CLASSES_MAP, REV_VOC_CLASSES_MAP

# NOTE: Write first the most important accuracy metric, and the
#       one returned FIRST by calling meter.value('all'). This is
#       important when either train/validate are called from
#       src/train_test.py.

# NOTE: Remember that value(metric='all') should always return a tuple


class ImageClassificationMeter(ClassErrorMeter):
    def __init__(self):
        self.metrics = ['top1', 'top5']
        super().__init__(topk=[1, 5], accuracy=True)

    def add(self, output, target):
        super().add(output.detach(), target)

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
        self.metrics = ['mIOU', 'AvgPixelAcc']
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.pixel_acc_sum = 0
        self.conf_mat = None
        self.num = 0

    def add(self, output, target):
        output = output['out'].detach()#.squeeze(0)
        target = target.squeeze(0).long()

        acc = self.pixel_acc(output, target)
        self.pixel_acc_sum += acc
        self.iou(output, target)
        self.num += 1

    def value(self, metric=None):
        assert metric in self.metrics + ['all'], f'Metric {metric} is not supported'
        
        avg_pixel_acc = self.pixel_acc_sum.item() / self.num
        if metric == 'AvgPixelAcc':
            return 100 * avg_pixel_acc

        # get iou for every class
        iou = torch.diag(self.conf_mat) / (self.conf_mat.sum(1) + self.conf_mat.sum(0) - torch.diag(self.conf_mat))
        m_iou = iou.mean().item()
        if metric == 'mIOU':
            return 100 * m_iou

        elif metric == 'all':
            return 100 * m_iou, 100 * avg_pixel_acc

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


class COCOMeter(Meter):
    """Accuracy meter for COCO detection
    metrics:
        mAP[IOU=0.5]
        mAP[IoU=0.50:0.95]
        mAP[IOU=0.75]
    """
    def __init__(self, coco_gt, iou_type="bbox"):
        self.metrics = ['mAP[IOU=0.5]', 'mAP[IoU=0.50:0.95]', 'mAP[IOU=0.75]']

        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_type = iou_type        
        self.reset()

    def reset(self):
        self.coco_eval = {}
        self.coco_eval[self.iou_type] = COCOeval(self.coco_gt, iouType=self.iou_type)

        self.img_ids = []
        self.eval_imgs = []

    def add(self, output, target):
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in output]
        predictions = {int(target['image_id']): outputs[0]}

        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        results = self.prepare_for_coco_detection(predictions)
        with redirect_stdout(io.StringIO()):
            coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
        coco_eval = self.coco_eval[self.iou_type]

        coco_eval.cocoDt = coco_dt
        coco_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = self.evaluate(coco_eval)

        self.eval_imgs.append(eval_imgs)

    def value(self, metric=None):
        assert metric in self.metrics + ['all'], f'Metric {metric} is not supported'
        
        eval_images = np.concatenate(self.eval_imgs, 2)
        self.gather_eval(self.coco_eval[self.iou_type], self.img_ids, eval_images)
        
        self.accumulate()
        self.summarize()

        mAP_05_09, mAP_05, mAP_075 = self.coco_eval["bbox"].stats[:3]

        if metric == 'all':
            return 100 * mAP_05, 100 * mAP_05_09, 100 * mAP_075
        elif metric == 'mAP[IoU=0.50:0.95]':
            return 100 * mAP_05_09

        elif metric == 'mAP[IOU=0.5]':
            return 100 * mAP_05

        elif metric == 'mAP[IOU=0.75]':
            return 100 * mAP_075

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            boxes = prediction["boxes"]
            boxes = self.convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def convert_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

    def evaluate(self, imgs):
        with redirect_stdout(io.StringIO()):
            imgs.evaluate()
        return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))

    def accumulate(self):
        self.coco_eval[self.iou_type].accumulate()

    def summarize(self):
        with redirect_stdout(io.StringIO()):
            self.coco_eval[self.iou_type].summarize()

    def gather_eval(self, coco_eval, img_ids, eval_imgs):
        eval_imgs = list(eval_imgs.flatten())
        coco_eval.evalImgs = eval_imgs
        coco_eval.params.imgIds = img_ids
        coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


class TranslationMeter(Meter):
    """Accuracy meter for machine translation
       metrics: BLEU score
    """
    def __init__(self):
        self.metrics = ['BLEU']
        self.reset()
        
    def reset(self):
        self.outputs = []
        self.targets = []
        self.num = 0
    
    def add(self, output, target):
        data, model, german, english, device

        targets = []
        outputs = []

        for example in data:
            src = vars(example)["src"]
            trg = vars(example)["trg"]

            prediction = translate_sentence(model, src, german, english, device)
            prediction = prediction[:-1]  # remove <eos> token

            targets.append([trg])
            outputs.append(prediction)
    
    
    def value(self, metric=None):
        self.bleu_score = self.bleu_score(self.outputs, self.targets)

        if metric == 'BLEU':
            return self.bleu_score     
        elif metric == 'all':
            return self.bleu_score,
