import logging
import os.path
import pickle
import pandas as pd
from time import time
from shutil import copy
from collections import namedtuple
from types import SimpleNamespace
from glob import glob
from src.accelerator_cfg import AcceleratorProfile
from src.scheduler import Scheduler, SchedulerType
from src.baseline import BaselineEvaluator
from src.optimizer import AcceleratorOptimizer as OurEvaluator


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
   theirs = ParitionEvaluator(args.partition_results_path)

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


PartitionPoint = namedtuple('PartitionPoint', ['points', 'accelerators',
                                               'overall_latency', 'overall_energy',
                                               'maximum_throughput',
                                               'overall_link_latency', 'overall_link_energy'])


class ParitionEvaluator(OurEvaluator):
   """Implementation of a parititioning-aware scheduling optimizer 
   """
   def __init__(self, workload, partition_results_path):
      self.load_partition_results(partition_results_path)
      
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

      metrics_evaluated = ['OverallLatency', 'OverallEnergy', 'MaximumThroughput', 'OverallLinkLatency', 'OverallLinkEnergy']

      files = glob(f'{path}/*/*result_nondom.csv')
      dfs = {os.path.basename(file).replace('result_nondom.csv', ''):
             pd.read_csv(file, header=None, index_col=0)
             for file in files}

      self.partitions = {}
      for arch, df in dfs.items():
         numeric_columns = [column for column in df.columns
                            if all(isnumber(entry) for entry in df.loc[:, column])]

         initial_columns = first_consecutive_numbers(numeric_columns)
         assert all(column in numeric_columns for column in initial_columns) and \
                max(initial_columns) + 1 not in numeric_columns

         partition_point_columns = first_consecutive_numbers([column for column in df.columns
                                                              if column not in numeric_columns])                
         metric_columns = numeric_columns[-len(metrics_evaluated):]
         partition_assignment_columns = [column for column in numeric_columns
                                         if column not in initial_columns and column not in metric_columns]
         
         self.partitions[arch] = [
            PartitionPoint(
               points=[row.loc[column] for column in partition_point_columns],
               accelerators=[row.loc[column] for column in partition_assignment_columns],
               **{metric: row.loc[column] for metric, column in zip(PartitionPoint._fields[2:], metric_columns)}
            )
            for row in df.iterrows()
         ]