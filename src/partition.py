import logging
import os.path
import pickle
import random
import re
import pandas as pd
from time import time
from shutil import copy
from collections import namedtuple, OrderedDict
from types import SimpleNamespace
from glob import glob
from src.accelerator_cfg import AcceleratorProfile
from src.scheduler import Scheduler, SchedulerType
from src.baseline import BaselineEvaluator
from src.args import MetricType


logger = logging.getLogger(__name__)


def run_partition_comparison(args, workload, accuracy_lut):
   """Experiment comparing a partition-aware scheduling technique
      for multi-DNN workloads with ours
   """
   accel_cfg = AcceleratorProfile(args.accelerator_arch_type)

   # evaluate our scheduling with a set architecture
   args.sceduler_type = SchedulerType.Ours
   ours = BaselineEvaluator(args, accel_cfg, workload, accuracy_lut)
   ours.evaluate()

   # a hypothetical partition-aware scheduling technique
   theirs = PartitionEvaluator(args, args.baseline_num_accelerators, workload)
   theirs.run_optimization()

   logger.info(f"Ours: evaluation results:\n"
            f"\tEnergy={ours.latest_energy:.3e}\n"
            f"\tLatency={ours.latest_latency:.3e}\n"
            f"\tEDP={ours.latest_energy * ours.latest_latency:.3e}\n"
            f"\tArea={ours.latest_area:.3e}")


PartitionMetrics = namedtuple('PartitionMetrics', ['partition_latency', 'partition_link_latency',
                                                   'overall_latency', 'overall_energy',
                                                   'maximum_throughput', 'overall_link_latency', 'overall_link_energy'])

PartitionInstance = namedtuple('PartitionInstance', ['tag', 'num_partitions', 'partition_points', 'assignment', 'metrics'])


class PartitionEvaluator:
   """Implementation of a parititioning-aware scheduling optimizer 
   """
   def __init__(self, args, num_accelerators, workload):
      self.logdir = args.logdir
      self.best_energy = self.best_edp = self.best_latency = None
      self.latest_energy = self.latest_edp = self.latest_latency = None
      self.best_schedule = self.latest_schedule = None
      self.best_partition = self.latest_partition = None
      self.latest_execution = self.latest_transfer = None

      self.optimization_metric = args.partition_optim_metric_type
      self.num_iterations = args.partition_optim_iterations
      self.selection_probability = args.partition_selection_probability
      self.num_accelerators = num_accelerators

      # initialize scheduler
      self.scheduler = Scheduler(SchedulerType.PartitionAware)
      
      # preapare partitioning data for scheduling
      self.load_partition_results(args.partition_results_path,
                                  list(workload.dnns.keys()))

      # determine type of optimization
      # TODO: This method is based on sample/evaluate iterations. Find a more sophisticated
      # TODO:  technique that takes into account all partitionings (not only sampled ones)
      self.optimize = self._optimize_search

   def load_partition_results(self, path, networks):
      """Load the partitioning results
      """
      def isnumber(entry):
         try:
            return float(entry) == entry
         except ValueError:
            return False

      def first_consecutive_numbers(number_list):
         consecutives = []
         for number in number_list:
            if number + 1 in number_list:
               consecutives.append(number)
               continue
            elif len(consecutives) != 0:
               consecutives.append(number)
               return consecutives
            
      def keep_uniques(seq):
         seen = set()
         seen_add = seen.add
         return [x for x in seq if not (x in seen or seen_add(x))]

      # import and load the csv files
      files = [file for file in glob(f'{path}/*/*result_nondom.csv')
               if re.search(f'/(\w+)_result_nondom[.]csv', file).group(1) in networks]
      logger.debug(f"Found {len(files)} csv files: ({' | '.join(files)})")

      dfs = OrderedDict([
         (os.path.basename(file).replace('_result_nondom.csv', ''),
          pd.read_csv(file, header=None, index_col=0))
         for file in files
      ])
      assert not any(df.empty for df in dfs.values()), \
         f"Empty result file(s) detected: {' | '.join([file for file in files if pd.read_csv(file).empty])}"

      self.partitions = OrderedDict()
      for arch, df in dfs.items():
         logger.debug(f'Parsing partition data for {arch}')

         # separate column indices in sections, not counting the index
         # first, the 2nd column shows the number of unique partition points
         num_unique_partition_points_column = 2
         # columns with numbers (i.e., not layer names)
         numeric_columns = [column for column in df.columns
                            if all(isnumber(entry) for entry in df.loc[:, column])]
         # columns with the names of partition points (i.e., layers)
         partition_point_columns = first_consecutive_numbers([column for column in df.columns
                                                              if column not in numeric_columns])
         total_num_partitions = len(partition_point_columns)
         # columns with the latency of each partition
         start_part_latency = num_unique_partition_points_column
         end_part_latency = start_part_latency + 1 + total_num_partitions
         partition_latency_columns = list(df.columns[start_part_latency:end_part_latency])
         # columns with the latency of the link between partitions
         start_part_link_latency = end_part_latency
         end_part_link_latency = start_part_link_latency + total_num_partitions
         partition_link_latency_columns = list(df.columns[start_part_link_latency:end_part_link_latency])
         # columns with the assignment of partitions to accelerators
         start_assignment = partition_point_columns[-1]
         end_assignment = start_assignment + total_num_partitions + 1
         partition_assignment_columns = list(df.columns[start_assignment:end_assignment])
         # columns with overall metrics
         overall_metric_columns = list(df.columns[end_assignment:-1])
         assert partition_latency_columns + partition_link_latency_columns + partition_point_columns + \
                partition_assignment_columns + overall_metric_columns == list(df.columns[num_unique_partition_points_column:-1]), arch
         assert len(partition_assignment_columns) == len(partition_latency_columns) == \
                len(partition_link_latency_columns) + 1 == len(partition_point_columns) + 1, arch

         # gather information about each group of partitions (one group per network)
         self.partitions[arch] = []
         for row_index, row in df.iterrows():

            this_num_partition_points = row.loc[num_unique_partition_points_column]
            if this_num_partition_points == 1:
               continue

            # TODO: The following code aims to keep only the non-identical partitions (signified by non-zero latency)
            #       However, the non-zero link latencies do not match the number of non-zero execution latencies
            #       Abandoning for now, but should fix this
            # assignments, partition_latency = zip(*[(row.loc[assignment], row.loc[latency]) for assignment, latency
            #                                        in zip(partition_assignment_columns, partition_latency_columns)
            #                                        if row.loc[latency] != 0])
            # assert len(assignments) == len(partition_latency), f"{arch} {row_index}"
            # partition_points, partition_link_latency = zip(*[
            #    (row.loc[partition_point], row.loc[link_latency])
            #    for partition_point, link_latency in zip(partition_point_columns, partition_link_latency_columns)
            #    if row.loc[link_latency] != 0
            # ])
            # assert len(partition_points) == len(partition_link_latency), f"{arch} {row_index}"

            assignments = [row.loc[column] for column in partition_assignment_columns]
            partition_points = [row.loc[column] for column in partition_point_columns]
            partition_latency = [row.loc[column] for column in partition_latency_columns]
            partition_link_latency = [row.loc[column] for column in partition_link_latency_columns]

            # overall schedule metrics
            overall_metrics = {metric: row.loc[column] for metric, column in 
                                zip(PartitionMetrics._fields[-len(overall_metric_columns):], overall_metric_columns)}

            # save each schedule as a namedtuple
            partition_instance = PartitionInstance(
               tag=f'{arch}_{row_index}',
               num_partitions=this_num_partition_points + 1,
               partition_points=partition_points,
               assignment=assignments,
               metrics=PartitionMetrics(
                  partition_latency=partition_latency,
                  partition_link_latency=partition_link_latency,
                  **overall_metrics)
            )
            self.partitions[arch].append(partition_instance)
            logger.debug(f'\t{row_index}: {partition_instance}')

   def save_results(self):
      """Save the most recent results
      """
      state_dict = {'best_partition': self.best_partition,
                    'best_schedule': self.best_schedule,
                    'best_energy': self.best_energy,
                    'best_latency': self.best_latency,
                    'best_edp': self.best_edp,}

      savefile = os.path.join(self.logdir, 'partition.results.pkl')
      with open(savefile, 'wb') as f:
         pickle.dump(state_dict, f)
      logger.info(f"Saved partition results in: {savefile}")

   def run_optimization(self):
      """Wrapper for optimization algorithm 
      """
      start = time()
      logger.info(f"=> Beginning partition-aware optimization")
      try:
         is_successful = self.optimize()
      finally:
         self.save_results()
      assert self.best_schedule is not None
      logger.info(f"Completed optimization in {time() - start:.3e}s")

      # log the results of the final scheduling
      logger.info("*--------------*")
      schedule_str = '\n\t'.join([f'{entry.tag} -> {entry.bin}' for entry in self.best_schedule.entries])
      logger.debug(f"Final scheduling:\n\t{schedule_str}")
      logger.info(f"Evaluation results:\n"
                  f"\tEnergy={self.best_energy:.3e}\n"
                  f"\tLatency={self.best_latency:.3e}\n"
                  f"\tEDP={self.best_energy * self.best_latency:.3e}\n")
      logger.info("*--------------*\n")

   def _optimize_search(self):
      """Search-based optimization algorithm to find optimal schedules for partitioned DNNs
      """
      for iteration in range(1, self.num_iterations + 1):
         logger.info("*--------------*")
         logger.info(f"Iteration {iteration}/{self.num_iterations}")
         start = time()

         self.sample_partitions()
         self.evaluate_partitions()

         # select optimization metric         
         latest_metric, best_metric = {
            MetricType.Energy: (self.latest_energy, self.best_energy),
            MetricType.Latency: (self.latest_latency, self.best_latency),
            MetricType.EDP: (self.latest_edp, self.best_edp)
         }.get(self.optimization_metric)
         assert latest_metric is not None, f"Optimization metric {self.optimization_metric} is not valid"

         # check whether to reject the sampled partitions
         if best_metric is None or latest_metric < best_metric:
            self.best_partition = self.latest_partition
            self.best_schedule = self.latest_schedule
            self.best_energy, self.best_latency, self.best_edp = self.latest_energy, self.latest_latency, self.latest_edp

            # save and log the results
            if best_metric is not None:
               logger.info("\033[92m" + f"New {self.optimization_metric.name.lower()} improvement " \
                           f"on iteration {iteration}: {best_metric:.3e} < {latest_metric:.3e}" + "\033[0m")
         else:
            logger.info(f"No improvement on iteration {iteration}: {best_metric:.3e} > {latest_metric:.3e}")

         logger.info(f'Completed iteration {iteration} in {time() - start:.3e}s')

   def sample_partitions(self):
      """Semi-random sampling of one partitioning instance from each network
      """
      self.latest_partition = OrderedDict()
      for arch, partition_instances in self.partitions.items():
         if self.best_partition is None or random.random() > self.selection_probability:
            self.latest_partition[arch] = random.choice(partition_instances)
         else:
            self.latest_partition[arch] = self.best_partition[arch]

      logstr = '\n\t'.join([f'{arch}: {partition_instance}' for arch, partition_instance in self.latest_partition.items()])
      logger.debug(f"New sampled partitions:\n\t{logstr}")

   def evaluate_partitions(self):
      """Run scheduling on the selected partition to gather evaluation metrics
      """
      bins = list(range(1, self.num_accelerators + 1))
      partitions = list(self.latest_partition.values())

      start = time()
      self.latest_schedule = self.scheduler.run(partitions, bins)
      logger.debug(f"Completed schedule in {time() - start:.3e}s")

      if self.latest_schedule is None:
         logger.warning(f'Could not find valid schedule')
         self.latest_energy = self.latest_latency = self.latest_edp = None
         return

      # make sure no errors in sub-partition assignment have occured
      assert all(
         all(
            entries[i].end <= entries[i+1].start
            for i in range(len(entries)-1)
         ) 
         for bin, entries in self.latest_schedule.as_dict('bin').items()
      )

      # gather metrics from the schedule
      self.latest_energy = sum([
         instance.metrics.overall_energy + instance.metrics.overall_link_energy
         for instance in self.latest_partition.values()
      ])
      self.latest_latency = max([
         entries[-1].end
         for bin, entries in self.latest_schedule.as_dict(main_key='bin').items()
      ])
      # latency (energy) is given in ms (mJ) instead of ns (uJ)
      self.latest_latency *= 1e3
      self.latest_energy *= 1e6
      self.latest_edp = self.latest_energy * self.latest_latency


if __name__ == "__main__":
   args = SimpleNamespace(logdir=None,
                          scheduler_type=None,
                          solver_type=None,
                          partition_optim_metric_type=None,
                          partition_optim_iterations=2,
                          partition_selection_probability=0.8,
                          partition_results_path='data/cnn-parted')
   e = PartitionEvaluator(args, num_accelerators=4)
   e.run_optimization()