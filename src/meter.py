from torchnet.meter import ClassErrorMeter

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
