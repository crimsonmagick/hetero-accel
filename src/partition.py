import logging
import os.path
import pickle
import random
import pandas as pd
from time import time
from shutil import copy
from collections import namedtuple, OrderedDict
from types import SimpleNamespace
from glob import glob
from src.accelerator_cfg import AcceleratorProfile
from src.scheduler import Scheduler
from src.baseline import BaselineEvaluator
from src.optimizer import AcceleratorOptimizer as OurEvaluator
from src.args import MetricType


logger = logging.getLogger(__name__)


def run_partition_comparison(args, workload, accuracy_lut):
   """Experiment comparing a partition-aware scheduling technique
      for multi-DNN workloads with ours
   """
   accel_cfg = AcceleratorProfile(args.accelerator_arch_type)
   accelerator = BaselineEvaluator(args, accel_cfg, workload, accuracy_lut).state

   logger.info("Accelerator for comparisons:")
   for sub_accelerator in accelerator:
      logger.info(f"\t{sub_accelerator}")

   # a hypothetical partition-aware scheduling technique
   theirs = PartitionEvaluator(args, args.baseline_num_accelerators)
   theirs.run_optimization()

   # our technique
   ours = OurEvaluator(args=args,
                       num_accelerators=len(accelerator),
                       accelerator_cfg=accel_cfg,
                       workload=workload,
                       accuracy_lut=accuracy_lut,
                       hw_constraints=SimpleNamespace(deadline=args.deadline_constraint,
                                                      area=args.area_constraint)
                       )
   ours.set_state(accelerator)
   ours.energy()


PartitionMetrics = namedtuple('PartitionMetrics', ['overall_latency', 'overall_energy',
                                                   'maximum_throughput', 'overall_link_latency', 'overall_link_energy'])

ParitionInstance = namedtuple('ParitionInstance', ['index', 'points', 'accelerators', 'metrics'])


class PartitionEvaluator:
   """Implementation of a parititioning-aware scheduling optimizer 
   """
   def __init__(self, args, num_accelerators):
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
      self.scheduler = Scheduler(args.scheduler_type)
      self.solver_type = args.solver_type
      
      # preapare partitioning data for scheduling
      self.load_partition_results(args.partition_results_path)

      # determine type of optimization
      # TODO: This method is based on sample/evaluate iterations. Find a more sophisticated
      # TODO:  technique that takes into account all partitionings (not only sampled ones)
      self.optimize = self._optimize_search

   def load_partition_results(self, path):
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

      files = glob(f'{path}/*/*result_nondom.csv')
      dfs = OrderedDict([
         (os.path.basename(file).replace('_result_nondom.csv', ''),
          pd.read_csv(file, header=None, index_col=0))
         for file in files
      ])
      assert not any(df.empty for df in dfs.values()), \
         f"Empty result file(s) detected: {' | '.join([file for file in files if pd.read_csv(file).empty])}"

      self.partitions = OrderedDict()
      for arch, df in dfs.items():

         # separate column indices in initial, partition points, assignment to accelerators and eval. metrics
         numeric_columns = [column for column in df.columns
                            if all(isnumber(entry) for entry in df.loc[:, column])]
         initial_columns = first_consecutive_numbers(numeric_columns)
         assert all(column in numeric_columns for column in initial_columns) and \
                max(initial_columns) + 1 not in numeric_columns
         partition_point_columns = first_consecutive_numbers([column for column in df.columns
                                                              if column not in numeric_columns])                
         metric_columns = numeric_columns[-len(PartitionMetrics._fields):]
         partition_assignment_columns = [column for column in numeric_columns
                                         if column not in initial_columns and column not in metric_columns]

         # gather information about each group of partitions (one group per network)
         self.partitions[arch] = [
            ParitionInstance(
               index=row_index,
               points=[row.loc[column] for column in partition_point_columns],
               accelerators=[row.loc[column] for column in partition_assignment_columns],
               metrics=PartitionMetrics(
                  **{metric: row.loc[column] 
                     for metric, column in zip(PartitionMetrics._fields, metric_columns)}
                  )
            )
            for row_index, row in df.iterrows()
         ]

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
      logger.info(f"Scheduler results:\n\t{schedule_str}")
      logger.info(f"Evaluation results:\n"
                  f"\tEnergy={self.best_energy:.3e}\n"
                  f"\tLatency={self.best_latency:.3e}\n"
                  f"\tEDP={self.best_energy * self.best_latency:.3e}\n")
      logger.info("*--------------*")

   def _optimize_search(self):
      """Search-based optimization algorithm to find optimal schedules for partitioned DNNs
      """
      for iteration in range(self.num_iterations):
         self.sample_partitions()
         self.prepare_mappings_from_partitions()
         self.evaluate_mappings()

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
            logger.info("\033[92m" + f"New {self.optimization_metric.name.lower()} improvement " \
                        f"on iteration {iteration}: {best_metric:.3e} -> {latest_metric:.3e}" + "\033[0m")
         else:
            logger.info(f"No improvement on iteration {iteration}: {best_metric:.3e} -> {latest_metric:.3e}")

   def sample_partitions(self):
      """Semi-random sampling of one partitioning instance from each network
      """
      self.latest_partition = OrderedDict()
      for arch, partition_instances in self.partitions.items():
         if self.best_partition is None or random.random() > self.selection_probability:
            self.latest_partition[arch] = random.choice(partition_instances)
         else:
            self.latest_partition[arch] = self.best_partition[arch]

   def prepare_mappings_from_partitions(self):
      """Gather possible partition-to-accelerator mappings
      """
      if self.latest_partition is None:
         self.sample_partitions()

      execution = OrderedDict()
      transfer = OrderedDict()

      # TODO: How to prepare this?

      for arch, partition in self.latest_partition.items():
         execution[arch] = OrderedDict()
         transfer[arch] = OrderedDict()

   def evaluate_mappings(self):
      """
      """
      self.scheduler.run()
      self.latest_partition
      self.latest_schedule
      self.latest_energy
      self.latest_latency
      self.latest_edp = self.latest_energy * self.latest_edp
      raise NotImplementedError



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