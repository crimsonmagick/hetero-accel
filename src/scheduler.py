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


__all__ = ['SchedulerType', 'ScheduleEntry', 'Schedule',
           'SolverType', 'solver_args_dict'
           'Scheduler']

logger = logging.getLogger(__name__)



class SchedulerType(Enum):
    Ours = 1
    Random = 2
    MultiKnapsack = 3
    Greedy = 4


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


# TODO: Study the possible options for the solver

class SolverType(Enum):
    Greedy = 1
    GreedyRegret = 2
    MTHGGreedy = 3
    MTHGGreedyRegret = 4
    LocalSearch = 5
    ColumnGenerationGreedy = 6
    ColumnGenerationLimitedDiscrepency = 7
    Random = 8
    LocalSolver = 9
    MixedIntegerLinearCBC = 10
    MixedIntegerLinearCPLEX = 11
    MixedIntegerLineargGurobi = 12
    MixedIntegerLinearKnitro = 13
    ConstraintGecode = 14
    ConstraintCPLEX = 15

solver_args_dict = {
    SolverType.Greedy: '\"greedy -f wij\"',
    SolverType.GreedyRegret: '\"greedy-regret -f wij\"',
    SolverType.MTHGGreedy: '\"mthg -f wij\"',
    SolverType.MTHGGreedyRegret: '\"mthg-regret -f wij\"',
    SolverType.LocalSearch: '\"local-search --threads 4\"',
    SolverType.ColumnGenerationGreedy: '\"column-generation-heuristic-greedy --linear-programming-solver cplex\"',
    SolverType.ColumnGenerationLimitedDiscrepency: '\"column-generation-heuristic-limited-discrepancy-search --linear-programming-solver cplex\"',
    SolverType.Random: 'random',
    SolverType.LocalSolver: 'localsolver',
    SolverType.MixedIntegerLinearCBC: 'milp-cbc',
    SolverType.MixedIntegerLinearCPLEX: 'milp-cplex',
    SolverType.MixedIntegerLineargGurobi: 'milp-gurobi',
    SolverType.MixedIntegerLinearKnitro: 'milp-knitro',
    SolverType.ConstraintGecode: 'constraint-programming-gecode',
    SolverType.ConstraintCPLEX: 'constraint-programming-cplex',
}


class Scheduler:
    """Implementations for the scheduler"""
    def __init__(self, scheduler_type=SchedulerType.Ours):
        self.type = scheduler_type
        self.__run_f = {
            SchedulerType.Ours: self._run_ours,
            SchedulerType.Random: self._run_random_scheduling,
            SchedulerType.MultiKnapsack: self._run_with_identical_bins,
            SchedulerType.Greedy: self._run_greedy,
        }.get(scheduler_type)

    def run(self, *args, **kwargs):
        """Wrapper over the main scheduling function, may be needed
        """
        if self.type == SchedulerType.Ours:
            if 'solver_type' not in kwargs:
                kwargs['solver_type'] = SolverType.MTHGGreedy
        return self.__run_f(*args, **kwargs)

    def _run_ours(self, items, bins, cost_dict, weight_dict, max_capacity=None,
                  solver_type=SolverType.MTHGGreedy, use_value=False):
        """Static scheduling with heterogeneous bins w.r.t. cost/value and weight per item,
           i.e., cost_dict and weight_dict have different values for different bins.
           This is an implementation of the generalized assignment problem. We use the solver
           algorithmic options found in: https://github.com/fontanf/generalizedassignmentsolver
        """
        # NOTE: Other options: Dynamic programming/Branch-and-bound?
        def write_input_file(infile):
            """Create the input file for the solver
            """
            with open(infile, 'w') as f:
                # write the number of bins (agents) and items (tasks)
                f.write(f'{len(bins)} {len(items)}\n')

                # write the cost or value/profit
                for bin in bins:
                    costs = []
                    for item in items:
                        # in the case of an invalid mapping, the cost/value does not matter
                        if weight_dict[(item, bin)] < 0:
                            costs.append(0)
                        # in the case of value/profit
                        elif use_value:
                            costs.append(
                                max([value for key, value in cost_dict.items() if item in key]) - cost_dict[(item, bin)]   
                            )
                        # in the case of cost
                        else:
                            costs.append(
                                cost_dict[(item, bin)]
                            )
                    # TODO: Check if casting to int here is a problem. Floats do not work for this solver
                    f.write(' '.join([str(int(cost)) for cost in costs]) + '\n')

                # write the weights
                for bin_idx, bin in enumerate(bins):
                    # negative weights are marked as invalid mappings, and are assigned higher than the
                    # maximum capacity of the bin, to make that mapping impossible
                    weights = [weight_dict[(item, bin)] if weight_dict[(item, bin)] > 0 else capacities[bin_idx] + 1
                               for item in items]
                    f.write(' '.join([str(int(weight)) for weight in weights]) + '\n')

                # write the maximum weight (capacity) of each bin (agent)
                f.write(' '.join([str(int(capacity)) for capacity in capacities]))

        schedule = Schedule(bins)
        # if no deadline cosntraint is provided, the sum of all possible assignments to each bin is used
        capacities = [
            sum([weight_dict[(item, bin)] for item in items]) if max_capacity is None else max_capacity
            for bin in bins
        ]

        solver_dir = os.path.join(project_dir, 'generalizedassignmentsolver')
        logdir = logging.getLogger().logdir
        resdir = os.path.join(logdir, 'scheduler_solver')
        os.makedirs(resdir, exist_ok=True)
        infile = os.path.join(resdir, 'inputs')
        outfile = os.path.join(resdir, 'output')
        solution_file = os.path.join(resdir, 'solution')
        logfile = os.path.join(resdir, 'log')

        # write cost/profit and weight to file
        write_input_file(infile)
        # construct command for solver
        solver_args = solver_args_dict.get(solver_type)    
        command = f"cd {solver_dir} && " \
                  f"./bazel-bin/generalizedassignmentsolver/main -v 3 " \
                  f"-a {solver_args} " \
                  f"-i {infile} -o {outfile} -c {solution_file}" \
                  f"2>&1 | tee {logfile}"
        logger.debug(f"GeneralizedAssignmentSolver command:\n{command}")

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
        return schedule

    def _run_random_scheduling(self, items, bins, value_dict, weight_dict):
        """Random assignment of items to bins
        """
        schedule = Schedule(bins)
        for item in items:
            bin_sel = random.choice(bins)
            while (item, bin_sel) not in value_dict:
                bin_sel = random.choice(bins)
            schedule.add(item, bin_sel, weight_dict[(item, bin_sel)])
        return schedule

    def _run_with_identical_bins(self):
        """Static scheduling with homogeneous bins w.r.t. of values and weights per item
           This is an implementation of the Multiple Knapsack problem
        """
        raise NotImplementedError

    def _run_greedy(self):
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
            sorted(cost_dict.items(), key=lambda item: item[1], reverse=True)
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
    def write_input_file(infile):
        """Create the input file for the solver
        """
        with open(infile, 'w') as f:
            # write the number of bins (agents) and items (tasks)
            f.write(f'{len(bins)} {len(items)}\n')

            # write the cost or value/profit
            for bin in bins:
                costs = []
                for item in items:
                    # in the case of an invalid mapping, the cost/value does not matter
                    if weight_dict[(item, bin)] < 0:
                        costs.append(0)
                    # in the case of value/profit
                    elif use_value:
                        costs.append(
                            max([value for key, value in cost_dict.items() if item in key]) - cost_dict[(item, bin)]   
                        )
                    # in the case of cost
                    else:
                        costs.append(
                            cost_dict[(item, bin)]
                        )
                # TODO: Check if casting to int here is a problem. Floats do not work for this solver
                f.write(' '.join([str(int(cost)) for cost in costs]) + '\n')

            # write the weights
            for bin_idx, bin in enumerate(bins):
                # negative weights are marked as invalid mappings, and are assigned higher than the
                # maximum capacity of the bin, to make that mapping impossible
                weights = [weight_dict[(item, bin)] if weight_dict[(item, bin)] > 0 else capacities[bin_idx] + 1
                            for item in items]
                f.write(' '.join([str(int(weight)) for weight in weights]) + '\n')

            # write the maximum weight (capacity) of each bin (agent)
            f.write(' '.join([str(int(capacity)) for capacity in capacities]))

    def execute_schedule():
        solver_dir = os.path.join(project_dir, 'generalizedassignmentsolver')
        write_input_file(f'{solver_dir}/gas.inp')
        solver_args = solver_args_dict.get(SolverType.MTHGGreedy)
        command = f"cd {solver_dir} && " \
                    f"./bazel-bin/generalizedassignmentsolver/main -v 3 " \
                    f"-a {solver_args} " \
                    f"-i gas.inp -o gas.out -c gas.solution " \
                    f"2>&1 | tee gas.log"
        # print("Command:", command)

        # run command
        start = time()
        p = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        # print(f"Executed solver command in {time() - start:.3e} with exitcode: {p.returncode}")

        # check if all items are assigned in the solution
        assert re.search('Number of items.*[(](\d+)%', p.stdout)
        if re.search('Number of items.*[(](\d+)%', p.stdout).group(1) != '100':
            print("Invalid assignment")
            return

        # get assignment via the stdout of the command
        schedule = Schedule(bins)
        assigns = re.search('Item\s+Agent\n.*?\n(.*)', p.stdout, re.DOTALL).group(1)
        assigns = [re.sub('\s+', ' ', assign.strip()).split(' ') for assign in assigns.split('\n')[:-1]]
        # complete the item-to-bin assignment and define the schedule
        for item_idx, bin_idx in assigns:
            item = items[int(item_idx)]
            bin = bins[int(bin_idx)]
            print(f"{item} ({cost_dict[(item, bin)]}, {weight_dict[(item, bin)]}) -> {bin}")
            schedule.add(item, bin, weight_dict[(item, bin)])

        # print(schedule.as_dict())
        # schedule.visualize('schedule.png')
        return schedule

    def get_results(schedule): 
        energy = sum([
            cost_dict[(entry.tag, entry.bin)] for entry in schedule.entries
        ])
        latency = max([
            sum([
                weight_dict[(entry.tag, entry.bin)] for entry in entries
            ]) for bin, entries in schedule.as_dict(main_key='bin').items()
        ])
        return energy, latency


    ########################
    import random
    random.seed(123)
    LOAD_STATE = True
    use_value = False

    if LOAD_STATE:
        import pickle
        with open('state.sa.pkl', 'rb') as f:
            state_dict = pickle.load(f)

        cost_dict = state_dict['energy']
        weight_dict = state_dict['latency']
        initial_state = state_dict['state']
        items = list({key[0] for key in weight_dict.keys()})
        bins = list({key[1] for key in weight_dict.keys()})
        capacities = [sum([weight_dict[(item, bin)] for item in items]) for bin in bins]

        print()
        schedule = execute_schedule()
        energy, latency = get_results(schedule)
        print("Energy:", energy)
        print("Latency:", latency)

    else:
        items = ['apple', 'banana', 'orange', 'nectarine', 'strawberry', 'mango']
        bins = ['BIN1', 'BIN2', 'BIN3']
        capacities = [100, 100, 100]
        
        cost_dict = {}
        weight_dict = {}

        for item_idx, item in enumerate(items):
            for bin_idx, bin in enumerate(bins):
                cost_dict[(item, bin)] = random.randint(1, 10)
                weight_dict[(item, bin)] = random.randint(1, 10)

        from copy import deepcopy
        og_cdict = deepcopy(cost_dict)
        og_wdict = deepcopy(weight_dict)

        prob = 0.0
        while prob < 1.1:
            print()
            print("Probability:", prob)
            cost_dict = deepcopy(og_cdict)
            weight_dict = deepcopy(og_wdict)

            for item_idx, item in enumerate(items):
                for bin_idx, bin in enumerate(bins):
                    if random.random() > prob:
                        cost_dict[(item, bin)] = 0#10*max([value for key, value in cost_dict.items()
                                                #   if item in key])
                        weight_dict[(item, bin)] = capacities[bin_idx] + 1
            print("Weight dict:", weight_dict)
            print("Value dict:", cost_dict)
            prob += 0.1

            schedule = execute_schedule()
            if schedule is None:
                continue
            energy, latency = get_results(schedule)
            print("Energy:", energy)
            print("Latency:", latency)
