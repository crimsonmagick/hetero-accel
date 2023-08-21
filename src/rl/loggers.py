import csv
import os
from src.rl import HW_Metrics


class CSVLogger:
    """Logging data to csv file"""
    def __init__(self, filename, headers):
        if not filename.endswith('.csv'):
            filename += '.csv'
        self.filename = filename
        self.headers = headers
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def write_record(self, record):
        with open(self.filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(record)


class EnvCSVLogger(CSVLogger):
    """CSV logger, tailored for logging RL environment statistics 
    """
    def __init__(self, log_dir=None, filename=None):
        # track only a single value: accuracy-related and accumulated (model-wide) HW metrics
        self.single_value_variables = ['episode', 'timestep', 'reward', 'top1', 'top5', 'loss']
        self.single_value_variables += ['total_' + metric for metric in HW_Metrics._fields]

        # track the entire history per episode: hardware-related proxy metrics and action history
        self.history_variables = list(HW_Metrics._fields)
        self.history_variables += ['action']

        if log_dir is None:
            log_dir = logging.getLogger().logdir
        # create the logging directory, if it doesn't exists
        elif not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if filename is None:
            filename = 'stats.csv'
        filename = os.path.join(log_dir, filename)
        super().__init__(filename, self.single_value_variables + self.history_variables)
        self.reset_history()

    def reset_history(self):
        """Reset the history of the tracked variables
        """
        self.history = {}
        for variable in self.headers:
            if variable in self.history_variables:
                self.history[variable] = []
            else:
                self.history[variable] = None

    def add_record(self, stats):
        """Append/Write the given variable values to the saved history
        """
        for key, value in stats.items():
            assert key in self.history, f"Variable {key} is not included in tracked variables" \
                                        f" ({' | '.join(self.history.keys())})"
            if key in self.history_variables:
                self.history[key].append(value)
            else:
                self.history[key] = value

    def log_stats(self):
        """Log all the tracked statistics to the csv file
        """
        record = []
        for value in self.history.values():
            if value is None or (isinstance(value, list) and len(value) < 1):
                value = ''
            record.append(value)
        self.write_record(record)

    def record_and_log_stats(self, stats):
        """Utility function that combines the above two functions
        """
        self.add_record(stats)
        self.log_stats()


