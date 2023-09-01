import logging
import traceback
import os.path
import wandb
import torch
import pickle
import numpy as np
from copy import deepcopy
from types import SimpleNamespace
from src import dataset_dirs, pretrained_checkpoint_paths
from src.utils import env_cfg, handle_model_subapps, lut2csv
from src.net_wrapper import TorchNetworkWrapper
from src.compression.compressor import PruningQuantizationCompressor
from src.dataset import load_data
from src.timeloop import TimeloopConfig
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


logger = logging.getLogger(__name__)


def main():
    """Main executing function
    """
    args = env_cfg()
    args.logdir = logging.getLogger().logdir

    # initialize DNNs and datasets
    models, datasets = setup_networks_datasets(args)

    # check if a DNN-accuracy LUT was specified
    if args.dnn_accuracy_lut_file is None or not os.path.exists(args.dnn_accuracy_lut_file):
        # otherwise, create it with an exhaustive exploration search
        args.dnn_accuracy_lut = pruning_quant_exploration(args, models, datasets)
        # save the lut using pickle
        with open(os.path.join(args.logdir, 'dnn_accuracy_lut.pkl'), 'wb') as f:
            pickle.dump(args.dnn_accuracy_lut, f)
    # if it was specified, load it using pickle
    else:
        with open(args.dnn_accuracy_lut_file, 'rb') as f:
            args.dnn_accuracy_lut = pickle.load(f)

    # write the LUT to a csv file
    lut2csv(args.dnn_accuracy_lut, args.logdir)

    # move to accelerator exploration phase
    accelerator_exploration(args)



def setup_networks_datasets(args):
    """Initialize the multi-DNN workload and the corresponding datasets
    """
    dnn_args = deepcopy(args)
    datasets = {}
    models = []
    for idx, arch in enumerate(args.arch):
        arch = arch.lower()
        dnn_args.arch = arch
 
        # configure dataset
        if 'cifar100' in arch and 'cifar100' not in datasets:
            dataset = 'cifar100'
        elif 'cifar10' in arch and 'cifar10' not in datasets:
            dataset = 'cifar10'
        elif 'imagenet' in arch and 'imagenet' not in datasets:
            dataset = 'imagenet'
        elif 'mnist' in arch and 'mnist' not in datasets:
            dataset = 'mnist'
        data_loaders = load_data(
            dataset, dataset_dirs[dataset], arch,
            args.batch_size, args.workers,
            args.validation_split,
            args.effective_train_size,
            args.effective_valid_size,
            args.effective_test_size,
            args.evaluate_model_mode,
            True, #self.args.verbose
        )
        datasets[dataset] = data_loaders

        # configure checkpoint
        dnn_args.dataset = dataset
        dnn_args.resumed_checkpoint_path = None
        if len(args.resumed_checkpoint_path) > idx:
            dnn_args.resumed_checkpoint_path = args.resumed_checkpoint_path[idx]
        elif args.pretrained and dataset != 'imagenet':
            dnn_args.resumed_checkpoint_path = pretrained_checkpoint_paths[arch]
            dnn_args.pretrained = False

        net_wrapper = TorchNetworkWrapper.from_args(dnn_args)
        models.append(net_wrapper.model)

        do_exit = handle_model_subapps(net_wrapper, data_loaders, args)

    if do_exit:
        exit(0)

    return models, datasets


def pruning_quant_exploration(args, models, datasets):
    """Exploration of possible pruning/quantization profiles
    """
    dnn_accuracy_lut = {}
    for model in models:
        dnn_accuracy_lut[model.arch] = {}
        logger.info(f'Beginning exhaustive exploration for {model.arch}')

        # initialize compressor
        compression_args = SimpleNamespace(logdir=args.logdir,
                                           pruning_high=args.pruning_high,
                                           pruning_low=args.pruning_low,
                                           quant_high=args.quant_high,
                                           quant_low=args.quant_low,
                                           layer_type_whitelist=(torch.nn.Conv2d,),
                                           pruning_group_type=args.pruning_group_type,
                                           timeloop_files=TimeloopConfig(args.accelerator_arch_type),
                                           # DNN args for inheritance from TorchNetworkWrapper
                                           profile_model=False,
                                           gpus=args.gpus,
                                           cpu=args.cpu,
                                           print_frequency=args.batch_print_frequency,
                                           verbose=args.model_verbose)
        compressor = PruningQuantizationCompressor(compression_args, datasets[model.dataset], model)

        # compute original statistics
        top1, top5, loss = compressor.validate() if args.use_validation_set else compressor.test()
        sparsity, size = compressor.compute_model_statistics()
        og_stats = {'top1': top1, 'top5': top5, 'loss': loss, 'sparsity': sparsity, 'size': size}
        dnn_accuracy_lut[model.arch]['original_statistics'] = og_stats
        og_stats_logstr = ', '.join([f'{metric.capitalize()}={value:.2f}' if metric != 'size' else
                                     f'{metric.capitalize()}={value:.2e}'
                                     for metric, value in og_stats.items()])
        logger.info(f'{model.arch}: Original statistics: {og_stats_logstr}')

        # iterate over all combinations for pruning and quantization
        compression_stats = {}
        best_sparsity = sparsity
        best_size = size
        for pruning_ratio in np.arange(args.pruning_low, min(1, args.pruning_high + args.pruning_incr), args.pruning_incr):
            for quant_bits in np.arange(args.quant_low, args.quant_high + 1, args.quant_incr):
                pruning_ratio = np.round(pruning_ratio, 2)
                logger.info(f'{model.arch}: Testing pruning of {100 * pruning_ratio}% and '
                            f'quantization of {quant_bits} bits')

                # reset the previous state of the network
                compressor.reset()
                # execute the compression profile
                compressor.prune_and_quantize(pruning_ratio, quant_bits)
                # evaluate for accuracy and network statistics
                top1, top5, loss = compressor.validate() if args.use_validation_set else compressor.test()
                sparsity, size = compressor.compute_model_statistics()
                stats = {'top1': top1, 'top5': top5, 'loss': loss, 'sparsity': sparsity, 'size': size}
                compression_stats[(pruning_ratio, quant_bits)] = stats
                stats_logstr = ', '.join([f'{metric.capitalize()}={value:.2f}' if metric != 'size' else
                                          f'{metric.capitalize()}={value:.2e}'
                                          for metric, value in stats.items()])
                logger.info(f'{model.arch}: Compressed statistics: {stats_logstr}')

                # if the accuracy constraints are satisfied, save the model
                if (args.top1_constraint is not None and top1 >= og_stats['top1'] - args.top1_constraint) or \
                   (args.top5_constraint is not None and top5 >= og_stats['top5'] - args.top5_constraint) or \
                   (args.loss_constraint is not None and loss <= og_stats['loss'] - args.top5_constraint):

                    # save/overwrite the model with highest sparsity or lowest memory size
                    if sparsity >= best_sparsity:
                        compressor.save_model(name=model.arch + '_best_sparsity')
                    if size <= best_size:
                        compressor.save_model(name=model.arch + '_best_size')

        dnn_accuracy_lut[model.arch]['compression_statistics'] = compression_stats

    return dnn_accuracy_lut


def accelerator_exploration(args):
    """Exploration to design/discover the sub-accelerator architectures
    """
    pass


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

