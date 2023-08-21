from collections import namedtuple


# HW measurements that describe the performance and properties of the DNN and the accelerator
HW_Metrics = namedtuple('HW_Metrics',
                        ['area', 'latency', 'power', 'energy', 'sparsity', 'size'])
# accuracy metrics
Accuracy_Metrics = namedtuple('Accuracy_Metrics',
                              ['top1', 'top5', 'loss'])

