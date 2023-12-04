import logging
import sys
from collections import OrderedDict, namedtuple
from enum import Enum


__all__ = ['SchedulerType', 'ScheduleEntry', 'Schedule', 'StaticScheduler']

logger = logging.getLogger(__name__)



class SchedulerType(Enum):
    Ours = 1
    Baseline = 2
    SoTA = 3


ScheduleEntry = namedtuple('ScheduleEntry',
                           ['start', 'end', 'bin', 'tag'])


class Schedule:
    def __init__(self, bins):
        self.bins = bins
        # the ending timestamp of the last inserted item for each bin
        self.end_timestamp = {bin: 0 for bin in bins}
        self.entries = []
        self.assigned = {}

    def add(self, item, to_bin, duration):
        """Add entry to the schedule
        """
        assert item not in self.assigned, f"Item {item} is already assigned to bin {self.assigned[item]}"
        start = self.end_timestamp[to_bin]
        end = start + duration
        self.end_timestamp[to_bin] = end
        self.entries.append(ScheduleEntry(start, end, to_bin, item))
        self.assigned[item] = to_bin
    
    def remove(self):
        raise NotImplementedError

    def violated_deadline(self, deadline):
        return any(end_timestamp >= deadline for end_timestamp in self.end_timestamp.values())

    def as_dict(self, main_key='bin'):
        assert main_key in ScheduleEntry._fields, f"Select as main_key one of the followning {ScheduleEntry._fields}"
        keys = sorted({getattr(entry, main_key) for entry in self.entries})
        return OrderedDict([
            (key, [entry for entry in self.entries if getattr(entry, main_key) == key])
            for key in keys
        ])

    def visualize(self, savefile=None):
        raise NotImplementedError


class StaticScheduler:
    """Implementation of a static scheduler"""
    def __init__(self, heterogeneous_bins=True):
        if heterogeneous_bins:
            self.run = self._run_w_different_bins
        else:
            self.run = self._run_w_identical_bins

        # ONLY for debugging
        if getattr(sys, 'gettrace', None) is not None and sys.gettrace():
            self.run = self._run_random_scheduling

    def _run_w_different_bins(self, items, bins, value_dict, weight_dict):
        """Static scheduling with heterogeneous bins w.r.t. value and weight per item,
           i.e., value_dict and weight_dict have different values for different bins.
        """
        schedule = Schedule(bins)
        # TODO: Write scheduling algorithm

    def _run_random_scheduling(self, items, bins, value_dict, weight_dict):
        schedule = Schedule(bins)
        for item in items:
            bin_sel = random.choice(bins)
            schedule.add(item, bin_sel, weight_dict[(item, bin_sel)])
            print(item, bin_sel)
            print(schedule.end_timestamp)
            print(schedule.entries)

        # return the schedule, organized per bin
        return schedule

    def _run_with_identical_bins(self, items, bins, value_dict, weight_dict):
        """Static scheduling with homogeneous bins w.r.t. of values and weights per item,
           i.e., value_dict and weight_dict have equal values for different bins.
           This is an implementation of the Multiple Knapsack problem
        """
        raise NotImplementedError


if __name__ == "__main__":
    import random

    s = StaticScheduler()
    items = ['apple', 'banana', 'orange', 'nectarine', 'strawberry']
    bins = ['BIN1', 'BIN2', 'BIN3']
    values = {}
    weights = {}

    for item in items:
        for bin in bins:
            values[(item, bin)] = random.randint(1, 10)
            weights[(item, bin)] = random.randint(1, 10)

    schedule = s.run(items, bins, values, weights)

    pass
