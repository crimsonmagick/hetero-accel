import logging
import sys

logger = logging.getLogger(__name__)

DEFAULT_REWARD = 0.0


def acc_size(accuracy_metrics, hw_metrics,
             original_accuracy_metrics, original_hw_metrics,
             accuracy_constraints, hw_constraints):
    if not accuracy_metrics.top1 >= original_accuracy_metrics.top1 - accuracy_constraints.top1 or \
       not hw_metrics.size >= original_hw_metrics.size - hw_constraints.size:
        return DEFAULT_REWARD

    return accuracy_metrics.top1 * hw_metrics.size

