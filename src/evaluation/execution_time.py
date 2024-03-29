import argparse
import re
import numpy as np
from datetime import datetime


def main():
    parser = argparse.ArgumentParser('Execution time arguments')
    parser.add_argument('--logfile', required=True, help='Logging file from ARTEMIS experiment')
    args = parser.parse_args()

    with open(args.logfile, 'r') as f:
        logfile = f.read()
    print(f"Logfile: {args.logfile}")

    timestamps = []
    for date, time in re.findall('([\d/]+) ([\d:]+) .*Move', logfile):
        timestamp = date + ' ' + time
        timestamp = datetime.strptime(timestamp, "%d/%m/%Y %H:%M:%S")
        timestamps.append(timestamp)
        
    delta = [(timestamps[i+1] - timestamps[i]).total_seconds()
             for i in range(len(timestamps[:-1]))]
    
    mean_epoch_time = np.mean(delta)
    print(f"Mean epoch time: {mean_epoch_time:5e}s")


if __name__ == '__main__':
    main()
