import logging

logger = logging.getLogger(__name__)


class MultiDNNScheduler:
    """Implementation of the Multi-DNN static scheduler"""
    def __init__(self, deadline_constraint):
        self.deadline = deadline_constraint

    def run(self, dnns, accelerators, energy_dict, latency_dict):
        # TODO: Write scheduling algorithm
        raise NotImplementedError

