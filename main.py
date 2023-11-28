import logging
import traceback
import os.path
import wandb
import torch
import pickle
import numpy as np
import pandas as pd
import yaml
from copy import deepcopy
from types import SimpleNamespace
from src import dataset_dirs
from src.workload import MultiDNNWorkload
from src.utils import env_cfg, handle_model_subapps
from src.net_wrapper import TorchNetworkWrapper
from src.compression.compressor import PruningQuantizationCompressor
from src.dataset import load_data
from src.accelerator_cfg import AcceleratorProfile
from src.optimizer import DesignSpace, AcceleratorOptimizer


logger = logging.getLogger(__name__)


def main():
    """Main executing function
    """
    args = env_cfg()
    args.logdir = logging.getLogger().logdir
    # save arguments as pkl, for reproducibility
    with open(os.path.join(args.logdir, 'args.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)

    # initialize the workload
    workload = setup_workload(args)

    # create a LUT of quantization profiles for each DNN-precision pairing
    dnn_accuracy_lut = quant_exploration(args, workload)

    # perform a DSE to define the sub-accelerator architectures
    accel, energy = accelerator_exploration(args, workload, dnn_accuracy_lut)


def setup_workload(args):
    """Initialize the multi-DNN workload
    """
    dnn_args = deepcopy(args)
    dnns = {}
    datasets = {}
    print_frequency = {}

    # load workload file
    with open(args.workload_cfg_file, 'r') as stream:
        workloads = yaml.safe_load(stream)
    multi_dnn_workload = workloads['workloads'][args.workload_idx]

    # setup each DNN separately
    for idx, workload_dict in enumerate(multi_dnn_workload):

        # configure and save DNN wrapper
        for name, value in workload_dict.items():
            setattr(dnn_args, name, value)
        net_wrapper = TorchNetworkWrapper.from_args(dnn_args)
        dnns[net_wrapper.model.arch] = net_wrapper
        print_frequency[net_wrapper.model.arch] = dnn_args.batch_print_frequency

        if dnn_args.dataset not in datasets:
            # configure dataset
            data_loaders = load_data(
                dnn_args.dataset,
                dataset_dirs[dnn_args.dataset],
                net_wrapper.model.arch,
                args.batch_size,
                args.workers,
                args.validation_split,
                args.effective_train_size,
                args.effective_valid_size,
                args.effective_test_size,
                args.evaluate_model_mode,
                True, #self.args.verbose
            )
            datasets[dnn_args.dataset] = data_loaders

        if handle_model_subapps(net_wrapper, data_loaders, args):
            exit(0)

    return MultiDNNWorkload(dnns, datasets, print_frequency)


def quant_exploration(args, workload):
    """Exploration of possible quantization profiles
    """
    skip_exploration = False
    if args.dnn_accuracy_lut_file is not None and os.path.exists(args.dnn_accuracy_lut_file):
        skip_exploration = True
        preloaded_dnn_accuracy_lut = pd.read_csv(args.dnn_accuracy_lut_file)
        logger.info(f'=> Skipping exhaustive exploration: loaded LUT from {args.dnn_accuracy_lut_file}')

    # structure of the LUT
    df = pd.DataFrame(columns=['Network', 'QuantBits', 'Top1', 'Top5', 'Loss', 'Sparsity', 'Size', 'Valid'])

    compressors = {}
    for arch, net_wrapper in workload.dnns.items():
        # initialize compressor
        compression_args = SimpleNamespace(logdir=args.logdir,
                                           pruning_high=args.pruning_high,
                                           pruning_low=args.pruning_low,
                                           quant_high=args.quant_high,
                                           quant_low=args.quant_low,
                                           layer_type_whitelist=args.layer_type_whitelist,
                                           pruning_group_type=args.pruning_group_type,
                                           accelerator_profile=AcceleratorProfile(args.accelerator_arch_type),
                                           # DNN args for inheritance from TorchNetworkWrapper
                                           profile_model=False,
                                           gpus=args.gpus,
                                           cpu=args.cpu,
                                           print_frequency=workload.print_frequency[arch],
                                           verbose=args.model_verbose)
        compressor = PruningQuantizationCompressor(compression_args,
                                                   workload.datasets[net_wrapper.model.dataset],
                                                   net_wrapper.model)
        compressors[arch] = compressor
        if skip_exploration:
            continue

        logger.info(f'=> Beginning exhaustive exploration for {arch}')

        # compute original statistics
        top1, top5, loss = compressor.validate() if args.use_validation_set else compressor.test()
        model_stats, _ = compressor.compute_model_statistics()
        og_stats = {'top1': top1, 'top5': top5, 'loss': loss,
                    'sparsity': model_stats['sparsity'], 'size': model_stats['size']}
        og_stats_logstr = ', '.join([f'{metric.capitalize()}={value:.2f}' if metric != 'size' else
                                     f'{metric.capitalize()}={value:.2e}'
                                     for metric, value in og_stats.items()])
        logger.info(f'{arch}: Original statistics: {og_stats_logstr}')
        # save the statistics to the LUT
        df.loc[len(df.index)] = ([arch, 32, top1, top5, loss, model_stats['sparsity'], model_stats['size'], 1])

        # iterate over quantization bits
        quant_bits_options = np.arange(args.quant_low, args.quant_high + 1, args.quant_incr)
        if 16 not in quant_bits_options:
            quant_bits_options = np.append(quant_bits_options, 16)

        for quant_bits in quant_bits_options:
            logger.info(f'{arch}: Testing quantization of {quant_bits} bits')

            # reset the previous state of the network
            compressor.reset()
            # execute the compression profile
            compressor.quantize(quant_bits)
            # evaluate for accuracy and network statistics
            top1, top5, loss = compressor.validate() if args.use_validation_set else compressor.test()
            model_stats, _ = compressor.compute_model_statistics()
            stats = {'top1': top1, 'top5': top5, 'loss': loss,
                     'sparsity': model_stats['sparsity'], 'size': model_stats['size']}
            stats_logstr = ', '.join([f'{metric.capitalize()}={value:.2f}' if metric != 'size' else
                                      f'{metric.capitalize()}={value:.2e}'
                                      for metric, value in stats.items()])
            logger.info(f'{arch}: Compressed statistics: {stats_logstr}')

            # binary flag whether the accuracy constraints are satisfied
            valid = 0
            if (args.top1_constraint is not None and top1 >= og_stats['top1'] - args.top1_constraint) or \
               (args.top5_constraint is not None and top5 >= og_stats['top5'] - args.top5_constraint) or \
               (args.loss_constraint is not None and loss <= og_stats['loss'] - args.top5_constraint):
                valid = 1

            # save the statistics to the LUT
            df.loc[len(df.index)] = ([arch, quant_bits, top1, top5, loss, model_stats['sparsity'], model_stats['size'], valid])

    if skip_exploration:
       df = preloaded_dnn_accuracy_lut

    # check if any valid solutions were found
    assert df['Valid'].sum() > 1, "No valid solutions were found, consider changing the compression settings or " \
                                  "loosen the accuracy constraints"

    # save LUT to .csv file
    df.to_csv(os.path.join(args.logdir, 'lut.csv'))

    return df


def accelerator_exploration(args, workload, accuracy_lut):
    """Exploration to design/discover the sub-accelerator architectures
    """
    precision_options = sorted(set(
        accuracy_lut.loc[accuracy_lut['Valid'] == 1]['QuantBits']
    ))
    # remove 32 bits from the options
    precision_options = precision_options[:-1]
    if 16 not in precision_options:
        precision_options.append(16)
    hw_constraints = SimpleNamespace(deadline=args.deadline_constraint,
                                     area=args.area_constraint)

    accel_cfg = AcceleratorProfile(args.accelerator_arch_type)
    design_space = DesignSpace(pe_array_x=accel_cfg.width_options,
                               pe_array_y=accel_cfg.height_options,
                               precision=precision_options,
                               sram_size=accel_cfg.sram_size_options,
                               ifmap_spad_size=accel_cfg.ifmap_spad_size_options,
                               weights_spad_size=accel_cfg.weights_spad_size_options,
                               psum_spad_size=accel_cfg.psum_spad_size_options)
    logger.debug(f"Examining design space: {design_space}")

    # initalize and run optimizer
    optimizer = AcceleratorOptimizer(simanneal_args=args,
                                     num_accelerators=len(precision_options),
                                     accelerator_cfg=accel_cfg,
                                     design_space=design_space,
                                     workload=workload,
                                     accuracy_lut=accuracy_lut,
                                     hw_constraints=hw_constraints)
    optimizer.run()

    return optimizer.best_state, optimizer.best_energy


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if logger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the logger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = logger.handlers
            logger.handlers = [h for h in logger.handlers if type(h) != logging.StreamHandler]
            logger.error(traceback.format_exc())
            logger.handlers = handlers_bak
        raise
    finally:
        if logger is not None and hasattr(logging.getLogger(), 'log_filename'):
            logger.info('')
            logger.info('Log file for this run: ' + os.path.realpath(logging.getLogger().log_filename))
            exit()

