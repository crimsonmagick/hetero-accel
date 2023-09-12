import logging
import traceback
import os.path
import wandb
import torch
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from types import SimpleNamespace
from src import dataset_dirs, pretrained_checkpoint_paths
from src.utils import env_cfg, handle_model_subapps, perfect_divisors
from src.net_wrapper import TorchNetworkWrapper
from src.compression.compressor import PruningQuantizationCompressor
from src.dataset import load_data
from src.accelerator_cfg import AcceleratorProfile
from src.rl import Design_Space
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


logger = logging.getLogger(__name__)


def main():
    """Main executing function
    """
    args = env_cfg()
    args.logdir = logging.getLogger().logdir
    # save arguments as pkl, for reproducibility
    with open(os.path.join(args.logdir, 'args.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)

    # initialize DNNs and datasets
    models, datasets = setup_networks_datasets(args)

    # create a LUT of compression profiles via exhaustive search
    dnn_accuracy_lut, compressors = pruning_quant_exploration(args, models, datasets)
    del models, datasets

    # move to accelerator exploration phase
    design_space = define_design_space(dnn_accuracy_lut, compressors)
    del compressors
    accelerator_exploration(args, design_space)


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

        if handle_model_subapps(net_wrapper, data_loaders, args):
            exit(0)

    return models, datasets


def pruning_quant_exploration(args, models, datasets):
    """Exploration of possible pruning/quantization profiles
    """
    skip_exploration = False
    if args.dnn_accuracy_lut_file is not None and os.path.exists(args.dnn_accuracy_lut_file):
        skip_exploration = True
        preloaded_dnn_accuracy_lut = pd.read_csv(args.dnn_accuracy_lut_file)
        logger.info(f'=> Skipping exhaustive exploration: loaded LUT from {args.dnn_accuracy_lut_file}')

    # structure of the LUT
    df = pd.DataFrame(columns=['Network', 'PruningGroup', 'PruningRatio', 'QuantBits',
                               'Top1', 'Top5', 'Loss', 'Sparsity', 'Size', 'Valid'])

    compressors = {}
    for model in models:
        # initialize compressor
        compression_args = SimpleNamespace(logdir=args.logdir,
                                           pruning_high=args.pruning_high,
                                           pruning_low=args.pruning_low,
                                           quant_high=args.quant_high,
                                           quant_low=args.quant_low,
                                           layer_type_whitelist=(torch.nn.Conv2d, torch.nn.Linear),
                                           pruning_group_type=args.pruning_group_type,
                                           accelerator_profile=AcceleratorProfile(args.accelerator_arch_type),
                                           # DNN args for inheritance from TorchNetworkWrapper
                                           profile_model=False,
                                           gpus=args.gpus,
                                           cpu=args.cpu,
                                           print_frequency=args.batch_print_frequency,
                                           verbose=args.model_verbose)
        compressor = PruningQuantizationCompressor(compression_args, datasets[model.dataset], model)
        compressors[model.arch] = compressor
        if skip_exploration:
            continue

        logger.info(f'=> Beginning exhaustive exploration for {model.arch}')

        # compute original statistics
        top1, top5, loss = compressor.validate() if args.use_validation_set else compressor.test()
        model_stats, _ = compressor.compute_model_statistics()
        og_stats = {'top1': top1, 'top5': top5, 'loss': loss,
                    'sparsity': model_stats['sparsity'], 'size': model_stats['size']}
        og_stats_logstr = ', '.join([f'{metric.capitalize()}={value:.2f}' if metric != 'size' else
                                     f'{metric.capitalize()}={value:.2e}'
                                     for metric, value in og_stats.items()])
        logger.info(f'{model.arch}: Original statistics: {og_stats_logstr}')
        # save the statistics to the LUT
        df.loc[len(df.index)] = ([model.arch, 'null', 0.0, 32, top1, top5, loss, sparsity, size, 1])

        # iterate over two modes of pruning: rows and columns, over all combinations of
        # pruning ratios and quantization bits
        best_sparsity = sparsity
        best_size = size
        for prune_mode in ['columns', 'rows']:
            compressor.pruner.set_pruning_mode(prune_mode)

            pruning_ratio = args.pruning_low
            while pruning_ratio <= args.pruning_high:
                for quant_bits in np.arange(args.quant_low, args.quant_high + 1, args.quant_incr):
                    logger.info(f'{model.arch}: Testing pruning {prune_mode} of {100 * pruning_ratio}% and '
                                f'quantization of {quant_bits} bits')

                    # reset the previous state of the network
                    compressor.reset()
                    # execute the compression profile
                    compressor.prune_and_quantize(pruning_ratio, quant_bits)
                    # evaluate for accuracy and network statistics
                    top1, top5, loss = compressor.validate() if args.use_validation_set else compressor.test()
                    model_stats, _ = compressor.compute_model_statistics()
                    stats = {'top1': top1, 'top5': top5, 'loss': loss,
                             'sparsity': model_stats['sparsity'], 'size': model_stats['size']}
                    stats_logstr = ', '.join([f'{metric.capitalize()}={value:.2f}' if metric != 'size' else
                                              f'{metric.capitalize()}={value:.2e}'
                                              for metric, value in stats.items()])
                    logger.info(f'{model.arch}: Compressed statistics: {stats_logstr}')

                    # binary flag whether the accuracy constraints are satisfied
                    valid = 0
                    if (args.top1_constraint is not None and top1 >= og_stats['top1'] - args.top1_constraint) or \
                       (args.top5_constraint is not None and top5 >= og_stats['top5'] - args.top5_constraint) or \
                       (args.loss_constraint is not None and loss <= og_stats['loss'] - args.top5_constraint):

                        valid = 1
                        # save/overwrite the model with highest sparsity or lowest memory size
                        if sparsity >= best_sparsity:
                            compressor.save_model(name=model.arch + '_best_sparsity')
                        if size <= best_size:
                            compressor.save_model(name=model.arch + '_best_size')

                    # save the statistics to the LUT
                    df.loc[len(df.index)] = ([model.arch, prune_mode, pruning_ratio, quant_bits,
                                              top1, top5, loss, sparsity, size, valid])

                # increment, to fix rounding numpy errors
                pruning_ratio = np.round(pruning_ratio + args.pruning_incr, 2)

    if skip_exploration:
       df = preloaded_dnn_accuracy_lut

    # check if any valid solutions were found
    assert df['Valid'].sum() > 1, "No valid solutions were found, consider changing the compression settings or " \
                                  "loosen the accuracy constraints"

    # save LUT to .csv file
    df.to_csv(os.path.join(args.logdir, 'lut.csv'))

    return df, compressors


def define_design_space(dnn_lut, compressors):
    """Define the design space constraints from the pruning/quantization results
    """
    def criterion(arch):
        valid = dnn_lut.loc[(dnn_lut['Valid'] == 1) & (dnn_lut['QuantBits'] != 32)]
        best = valid.loc[valid['Size'] == min(valid['Size'])]
        return best['PruningGroup'].iloc[0], best['PruningRatio'].iloc[0], best['QuantBits'].iloc[0]

    ds_options = {'rows': [], 'columns': [], 'size': []}
    for arch, compressor in compressors.items():
        # find optimal compression profile
        pruning_mode, pruning_ratio, quant_bits = criterion(arch)
        # apply pruning and quantization
        compressor.reset()
        compressor.pruner.set_pruning_mode(pruning_mode)
        compressor.prune_and_quantize(pruning_ratio, quant_bits)

        # get the statistics per layer, including block-wise sparsity
        total_stats, layer_stats = compressor.compute_model_statistics()

        ds_options[pruning_mode] += [_layer_stats['nonzero_' + pruning_mode]
                                     for _layer_stats in layer_stats.values()]
        ds_options['size'] += [_layer_stats['size'] for _layer_stats in layer_stats.values()]

    pe_array_x = sorted(set(perfect_divisors(ds_options['columns'])))
    pe_array_y = sorted(set(perfect_divisors(ds_options['rows'])))
    if not pe_array_x:
        pe_array_x = pe_array_y
    elif not pe_array_y:
        pe_array_y = pe_array_x
    size = sorted(set(perfect_divisors(ds_options['size'])))

    # TODO: Consider how to limit the options per buffer, depending on its type and stored tensors

    return Design_Space(pe_array_x=pe_array_x,
                        pe_array_y=pe_array_y,
                        sram_size=size,
                        ifmap_spad_size=size,
                        weights_spad_size=size,
                        psum_spad_size=size)


def accelerator_exploration(args, design_space):
    """Exploration to design/discover the sub-accelerator architectures
    """
    # initialize agent
    agent_args = SimpleNamespace(logdir=args.logdir,
                                 prefix=None,
                                 deterministic=args.rl_agent_deterministic,
                                 seed=args.global_seed,
                                 verbose=args.rl_agent_verbose,
                                 policy_kwargs=dict(),
                                 device=env.compressor.device,
                                 eval_frequency=args.rl_agent_eval_frequency,
                                 no_improv_evals=args.rl_agent_no_improv_evals,
                                 min_evals=args.rl_agent_min_evals,
                                 timesteps=args.rl_agent_total_timesteps,
                                 train_episodes=args.rl_agent_train_episodes,
                                 eval_episodes=args.rl_agent_eval_episodes,
                                 load_from=args.rl_agent_load_from_path)
    agent = A2C_Agent(agent_args, env, None)

    # train the agent
    agent.learn()

    # evaluate the agent
    eval_env = env
    mean_reward, std_reward = agent.evaluate_policy(eval_env)
    if mean_reward is not None and std_reward is not None:
        logger.info("Policy evaluation: mean rewards: {mean_reward:.3e}, std rewards: {std_reward:.3e}")

    # explicit episode execution to gather final actions and metrics
    obs_t0 = eval_env.env.reset()
    action, _ = agent.predict(obs)
    obs_t1, reward, done, info = env.step(action)

    agent.finalize()


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

