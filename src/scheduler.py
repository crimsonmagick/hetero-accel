import logging
import os.path
import math
import re
import subprocess
import numpy as np
from collections import OrderedDict, namedtuple
from enum import Enum
from time import time
# matplotlib imports
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from matplotlib import pyplot as plt
from matplotlib import rcParams
from src import project_dir


__all__ = ['SchedulerType', 'ScheduleEntry', 'Schedule', 'StaticScheduler']

logger = logging.getLogger(__name__)



class SchedulerType(Enum):
    Ours = 1
    Random = 2
    MultiKnapsack = 3
    Exhaustive = 4
    Greedy = 5


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
        self.entries.append(ScheduleEntry(start, end, bin=to_bin, tag=item))
        self.assigned[item] = to_bin

    def as_dict(self, main_key='bin'):
        assert main_key in ScheduleEntry._fields, f"Select as main_key one of the followning {ScheduleEntry._fields}"
        keys = sorted({getattr(entry, main_key) for entry in self.entries})
        return OrderedDict([
            (key, [entry for entry in self.entries if getattr(entry, main_key) == key])
            for key in keys
        ])

    def visualize(self, savefile=None):
        """Visualize the current schedule
        """
        # rcParams.update({
        #     # 'text.latex.preamble': r"\usepackage{lmodern}",
        #     'font.size': "7",    
        #     "text.usetex": True,
        #     "font.family": "lmodern",
        #     "font.serif": ["Computer Modern Roman"]
        # })

        # organize schedule in batches: each batch contains one item per bin, and
        # there are as many batches as max number of items in one bin
        bin_dict = self.as_dict('bin')
        batch_entries = []
        batch_index = 0
        last_added = True
        while last_added:
            this_batch = []
            last_added = False
            for bin, bin_entries in bin_dict.items():
                try:
                    entry_for_this_batch = bin_entries[batch_index]
                    this_batch.append(entry_for_this_batch)
                    last_added = True
                except IndexError:
                    this_batch.append(ScheduleEntry(0, 0, bin, tag=''))

            if last_added:
                batch_entries.append(this_batch)
                batch_index += 1

        # build schedule figure
        fig, ax = plt.subplots(figsize=(3.3, 1.8))
        left = np.zeros(len(bin_dict))
        for batch_entry in batch_entries:
            widths = [entry.end - entry.start for entry in batch_entry]
            y = [repr(bin) for bin in bin_dict.keys()]

            bar_container = ax.barh(y=y,
                                    width=widths,
                                    height=0.9,
                                    align='center',
                                    left=left,
                                    joinstyle='round',
                                    capstyle='round',
                                    fill=False,
                                    linewidth=1.0,
                                    edgecolor='black',)
            ax.bar_label(bar_container,
                         labels=[f'{entry.tag}\n{entry.start}->{entry.end}'
                                  for entry in batch_entry],
                         label_type='center',
                         fontsize='x-small',
                         color='black')
            left += widths

        # save schedule figure
        if savefile is None:
            logdir = logging.getLogger().logdir
            savefile = os.path.join(logdir, 'latest_schedule.png')
        plt.savefig(savefile, bbox_inches='tight', pad_inches=0)
        logger.info(f"Schedule was saved in {savefile}")


# TODO: Configure scheduler to work without complete value/weight dicts

class Scheduler:
    """Implementations for the scheduler"""
    def __init__(self, scheduler_type=SchedulerType.Ours):
        self.type = scheduler_type
        self.run = {
            SchedulerType.Ours: self._run_ours,
            SchedulerType.Random: self._run_random_scheduling,
            SchedulerType.MultiKnapsack: self._run_with_identical_bins,
            SchedulerType.Exhaustive: self._run_exhaustive,
            SchedulerType.Greedy: self._run_greedy,
        }.get(scheduler_type)

        # # ONLY for debugging
        # if getattr(sys, 'gettrace', None) is not None and sys.gettrace():
        #     self.run = self._run_random_scheduling

    def _run_ours(self, items, bins, value_dict, weight_dict, max_capacity=None):
        """Static scheduling with heterogeneous bins w.r.t. value and weight per item,
           i.e., value_dict and weight_dict have different values for different bins.
           This is an implementation of the generalized assignment problem
           We use the solver algorithmic options found in: https://github.com/fontanf/generalizedassignmentsolver
        """
        # NOTE: Other options: Dynamic programming/Branch-and-bound?
        def write_input_file(infile):
            """Create the input file for the solver
            """
            with open(infile, 'w') as f:
                # write the number of bins (agents) and items (tasks)
                f.write(f'{len(bins)} {len(items)}\n')
                # write the value/profit (inverse of the cost)
                for bin in bins:
                    profits = [
                        max([value for key, value in value_dict.items() if item in key]) - value_dict[(item, bin)]
                        for item in items
                    ]
                    # TODO: Check if casting to int here is a problem. Floats do not work for this solver
                    f.write(' '.join([str(int(profit)) for profit in profits]) + '\n')
                # write the weights
                for bin in bins:
                    weights = [weight_dict[(item, bin)] for item in items]
                    f.write(' '.join([str(int(weight)) for weight in weights]) + '\n')
                # write the maximum weight (capacity) of each bin (agent)
                f.write(' '.join([str(int(capacity)) for capacity in capacities]))

        schedule = Schedule(bins)
        # TODO: This could be the deadline cosntraint, right now is the sum of all possible assignments
        capacities = [
            sum([weight_dict[(item, bin)] for item in items]) if max_capacity is None else max_capacity
            for bin in bins
        ]

        solver_dir = os.path.join(project_dir, 'generalizedassignmentsolver')
        logdir = logging.getLogger().logdir
        resdir = os.path.join(logdir, 'generalizedassignmentsolver')
        os.makedirs(resdir, exist_ok=True)
        infile = os.path.join(resdir, 'inputs')
        outfile = os.path.join(resdir, 'output')
        solution_file = os.path.join(resdir, 'solution')
        logfile = os.path.join(resdir, 'log')

        # write value/profit (oposite to cost) and weight to file
        write_input_file(infile)
        # construct command for solver
        # TODO: Study the possible options for the solver
        command = f"cd {solver_dir} && " \
                  f"./bazel-bin/generalizedassignmentsolver/main -v 3 " \
                  f"-a 'mthg -f -pij/wij' " \
                  f"-i {infile} -o {outfile} -c {solution_file} " \
                  f"2>&1 | tee {logfile}"
        logger.debug(f"Scheduling generalizedassignmentsolver command:\n{command}")

        # run command
        start = time()
        p = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.debug(f"Executed solver command in {time() - start:.3e} with exitcode: {p.returncode}")

        # check if all items are assigned in the solution
        if re.search('Number of items.*[(](\d+)%', p.stdout).group(1) != '100':
            return

        # get assignment via the stdout of the command
        assigns = re.search('Item\s+Agent\n.*?\n(.*)', p.stdout, re.DOTALL).group(1)
        assigns = [re.sub('\s+', ' ', assign.strip()).split(' ') for assign in assigns.split('\n')[:-1]]
        # complete the item-to-bin assignment and define the schedule
        for item_idx, bin_idx in assigns:
            item = items[int(item_idx)]
            bin = bins[int(bin_idx)]
            schedule.add(item, bin, weight_dict[(item, bin)])

        # # read the solution which contains an ordered list of bins, per item
        # with open(solution_file, 'r') as f:
        #     solution = f.read()
        # # complete the item-to-bin assignment and define the schedule
        # for item, bin_idx in zip(items, solution.strip().replace('\n', ' ').split(' ')):
        #     schedule.add(item, bins[bin_idx], weight_dict[(item, bins[bin_idx])])

        return schedule

    def _run_random_scheduling(self, items, bins, value_dict, weight_dict, **kwargs):
        """Random assignment of items to bins
        """
        schedule = Schedule(bins)
        for item in items:
            bin_sel = random.choice(bins)
            while (item, bin_sel) not in value_dict:
                bin_sel = random.choice(bins)
            schedule.add(item, bin_sel, weight_dict[(item, bin_sel)])
        return schedule

    def _run_with_identical_bins(self, items, bins, value_dict, weight_dict, **kwargs):
        """Static scheduling with homogeneous bins w.r.t. of values and weights per item,
           i.e., value_dict and weight_dict have equal values for different bins.
           This is an implementation of the Multiple Knapsack problem
        """
        raise NotImplementedError

    def _run_exhaustive(self, items, bins, value_dict, weight_dict, **kwargs):
        """Exhaustive search for optimal static scheduling to maximize 
        """
        raise NotImplementedError

    def _run_greedy(self, items, bins, value_dict, weight_dict, **kwargs):
        """Greedy implementation of a static scheduling for the generative
           assignment problem. Options here are:
           1) Order the items based on their value and then,
              at each timestep assign the task with the greater difference between
              their first and second-best assignment value.
           2) Order the items based on their value and start the assignment
              from highest to lowest
        """
        raise NotImplementedError("Not yet completed implementation")

        schedule = Schedule(bins)

        # order all items based on their value for any bin
        ordered_value_dict = OrderedDict(
            sorted(value_dict.items(), key=lambda item: item[1], reverse=True)
        )

        # 2d array of differences between all values for each item and any bin
        diff_2d = {}
        for item in items:
            values_2d = np.array(
                [value for key,value in ordered_value_dict.items() if item in key]
            )
            values_2d = np.expand_dims(values_2d, 1)
            diff_2d[item] = values_2d - values_2d.transpose() 

        items_assigned = []
        while len(items_assigned) < len(items):
            select_index = 1

            item_bin_pair, _ = ordered_value_dict.popitem(select_index)
            schedule.add(item_bin_pair[0], item_bin_pair[1], weight_dict[item_bin_pair])
            items_assigned.append(item_bin_pair[0])




if __name__ == "__main__":
    import random

    items = ['apple', 'banana', 'orange', 'nectarine', 'strawberry']
    bins = ['BIN1', 'BIN2', 'BIN3']
    capacities = [100, 100, 100]
    value_dict = {}
    weight_dict = {}
    schedule = Schedule(bins)

    for item in items:
        for bin in bins:
            value_dict[(item, bin)] = random.randint(1, 10)
            weight_dict[(item, bin)] = random.randint(1, 10)

    solver_dir = os.path.join(project_dir, 'generalizedassignmentsolver')
    with open(f'{solver_dir}/gas.inp', 'w') as f:
        f.write(f'{len(bins)} {len(items)}\n')
        # write the value/profit
        for bin in bins:
            profits = [max([value for key, value in value_dict.items() if item in key]) - value_dict[(item, bin)]
                       for item in items]
            f.write(' '.join([str(int(profit)) for profit in profits]) + '\n')

        for bin in bins:
            weights = [weight_dict[(item, bin)] for item in items]
            f.write(' '.join([str(int(weight)) for weight in weights]) + '\n')

        f.write(' '.join([str(int(capacity)) for capacity in capacities]))


    command = f"cd {solver_dir} && " \
                f"./bazel-bin/generalizedassignmentsolver/main -v 3 " \
                f"-a 'mthg -f -pij/wij' " \
                f"-i gas.inp -o gas.out -c gas.solution " \
                f"2>&1 | tee gas.log"

    # run command
    start = time()
    p = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    print(f"Executed solver command in {time() - start:.3e} with exitcode: {p.returncode}")

    # check if all items are assigned in the solution
    if re.search('Number of items.*[(](\d+)%', p.stdout).group(1) != '100':
        exit(1)

    # get assignment via the stdout of the command
    assigns = re.search('Item\s+Agent\n.*?\n(.*)', p.stdout, re.DOTALL).group(1)
    assigns = [re.sub('\s+', ' ', assign.strip()).split(' ') for assign in assigns.split('\n')[:-1]]
    # complete the item-to-bin assignment and define the schedule
    for item_idx, bin_idx in assigns:
        item = items[int(item_idx)]
        bin = bins[int(bin_idx)]
        schedule.add(item, bin, weight_dict[(item, bin)])

    print(schedule.as_dict())
    schedule.visualize('schedule.png')
