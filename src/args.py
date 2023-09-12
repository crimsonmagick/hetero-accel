import argparse
import os
import numpy as np
from enum import Enum
from src import project_dir
from src.accelerator_cfg import AcceleratorType
from src.rl import reward as rewards
from src.compression.pruning import PruningGroupType


def app_args(parser):
    """Arguments for running the main application"""
    parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs',
                         help='Path to dump logs and checkpoints')
    parser.add_argument('--name', '-n', help='Experiment name')
    parser.add_argument('--deterministic', '--det', action='store_true',
                        help='Set for deterministic execution. Preferably, set the experiment --seed as well')
    parser.add_argument('--seed', dest='global_seed', type=int,
                        help='Global seed for deterministic execution')
    parser.add_argument('--yaml-cfg-file', 
                        help='YAML file containing the experiment description')
    parser.add_argument('--dnn-accuracy-lut-file', metavar='PATH',
                        help='Path where the DNN-accuracy LUT is saved. Should contain '
                             'statistics for each DNN, per pruning-quantization profile')

    op_mode = parser.add_argument_group("Execution mode arguments")
    op_mode_exc = op_mode.add_mutually_exclusive_group()
    op_mode_exc.add_argument('--evaluate-model', dest='evaluate_model_mode', action='store_true',
                             help='Evaluate DNN model on test set')
    op_mode_exc.add_argument('--train-model', dest='train_model_mode', action='store_true',
                             help='Train DNN model on train set')
    op_mode_exc.add_argument('--test-timeloop-accelergy', dest='test_timeloop_accelergy_mode', action='store_true',
                             help='Execute an example to test if timeloop+accelergy works well')
    op_mode_exc.add_argument('--model-summary', dest='model_summary_mode', type=model_summary_type_arg, default='',
                             help='Specify the type of summary of the given DNNs')
    op_mode_exc.add_argument('--test-pruning-quantization', dest='test_pruning_quant_mode', action='store_true',
                             help='Execute a test of the effect of pruning on energy consumption')
    return parser


def model_args(parser):
    """Arguments for configuring the application model"""
    model_args = parser.add_argument_group("DNN related arguments")
    model_args.add_argument('-a', '--arch', nargs='+', default=[], help='Define NN models.')
    model_args.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='Number of data loading workers (default: 4)')
    model_args.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                            help='Mini-batch size (default: 256)')
    model_args.add_argument('--load-serialized', dest='load_serialized', action='store_true', default=False,
                            help='Load a model without DataParallel wrapping it')
    model_args.add_argument('--gpus', metavar='DEV_ID', nargs='*',
                            help='List of GPU device IDs to be used. Can be used as space- or comma-separated. '
                                 'Default is to use all available devices.')
    model_args.add_argument('--cpu', action='store_true', default=False,
                            help='Use CPU only. \n'
                                 'Flag not set => uses GPUs according to the --gpus flag value.'
                                 'Flag set => overrides the --gpus flag')
    model_args.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split', type=float, default=0.1,
                            help='Portion of training dataset to set aside for validation')
    model_args.add_argument('--effective-train-size', '--etrs', type=float, default=1.,
                            help='Portion of training dataset to be used in each epoch. '
                                 'NOTE: If --validation-split is set, then the value of this argument is applied '
                                 'AFTER the train-validation split according to that argument')
    model_args.add_argument('--effective-valid-size', '--evs', type=float, default=1.,
                            help='Portion of validation dataset to be used in each epoch. '
                                 'NOTE: If --validation-split is set, then the value of this argument is applied '
                                 'AFTER the train-validation split according to that argument')
    model_args.add_argument('--effective-test-size', '--etes', type=float, default=1.,
                            help='Portion of test dataset to be used in each epoch')
    model_args.add_argument('--batch-print-frequency', '--bpf', default=10, type=int, dest='batch_print_frequency',
                            help='Printing frequency for data batches (default: every 10 batches)')
    model_args.add_argument('--model-verbose', '--mv', action='store_true', 
                            help='Log various messages for the application model')
    model_args.add_argument('--model-save-frequency', type=int, dest='model_save_frequency',
                            help='Model saving frequency, measured in epochs. Default is to save only at the best epoch')
    model_args.add_argument('--use-profiler', action='store_true',
                            help='Use the tensorboard profiler to log model statistics')
    model_args.add_argument('--train-epochs', type=int, default=1,
                            help='Number of training epochs. Used only in training mode (with --train argument). '
                                 'Default is 1 epoch.')

    load_checkpoint_group = parser.add_argument_group('Resuming arguments')
    load_checkpoint_group_exc = load_checkpoint_group.add_mutually_exclusive_group()
    load_checkpoint_group_exc.add_argument('--resume-from', dest='resumed_checkpoint_path', nargs='+', default=[],
                                           help='Path(s) to latest checkpoint. Use to resume paused training session.')
    load_checkpoint_group_exc.add_argument('--pretrained', action='store_true', help='Use pre-trained models')

    optimizer_args = parser.add_argument_group('Optimizer arguments')
    optimizer_args.add_argument('--optimizer-type', '--ot', type=optimizer_type_arg, default='sgd',
                                help='Choose optimizer type')
    optimizer_args.add_argument('--learning-rate', '--lr', default=0.1, type=float, help='Initial learning rate (default: 0.1)')
    optimizer_args.add_argument('--momentum', default=0.9, type=float, help='Momentum (default: 0.9)')
    optimizer_args.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='Weight decay (default: 1e-4)')

    return parser


def compression_args(parser):
    """Arguments for pruning/quantization exploration
    """
    compression_args = parser.add_argument_group("Compression related arguments")
    compression_args.add_argument('--pruning-group-type', type=pruning_group_type_arg, default='columns',
                                  help='Pruning group type. Default is column pruning')
    compression_args.add_argument('--use-validation-set', action='store_true',
                                  help='Whether to use validation set during inference. If not specified, test set is used')
    compression_args.add_argument('--pruning-high', type=float, default=0.95,
                                  help='Highest bound for pruning')
    compression_args.add_argument('--pruning-low', type=float, default=0.0,
                                  help='Lowest bound for pruning')
    compression_args.add_argument('--pruning-increment', '--pruning-incr', dest='pruning_incr', type=float, default=0.05,
                                  help='Increment step for pruning exploration')
    compression_args.add_argument('--quant-high', type=int, default=8,
                                  help='Highest bound for weight quantization')
    compression_args.add_argument('--quant-low', type=int, default=2,
                                  help='Lowest bound for weight quantization')
    compression_args.add_argument('--quantization-increment', '--quant-incr', dest='quant_incr', type=int, default=2,
                                  help='Increment step for quantization exploration')
    compression_args.add_argument('--top1-constraint', type=float, default=1,
                                  help='Top1 accuracy loss constraint (default is 1\%)')
    compression_args.add_argument('--top5-constraint', type=float,
                                  help='Top5 accuracy loss constraint')
    compression_args.add_argument('--loss-constraint', type=float,
                                  help='Objective loss constraint')

    return parser


def rl_args(parser):
    """Arguments for reinforcement learning
    """
    ALL_REWARDS = sorted(reward_name for reward_name in rewards.__dict__
                         if reward_name.islower() and not reward_name.startswith('__')
                         and callable(rewards.__dict__[reward_name]))

    rl_args = parser.add_argument_group("RL related arguments")
    rl_args.add_argument('--rl-retrain-epochs', type=int, default=0,
                         help='Retraining epochs for accuracy exploration')
    rl_args.add_argument('--rl-model-save-frequency', type=int, default=1,
                         help='Model saving frequency, measured in episodes. Default is every episode')
    rl_args.add_argument('--rl-reward-type', choices=ALL_REWARDS,
                         help=f'Type of reward for the RL agent. Choices: {ALL_REWARDS}')
    rl_args.add_argument('--rl-energy-constraint', type=float, default=10,
                         help='Energy constraint (default is 10\%)')
    rl_args.add_argument('--rl-sparsity-constraint', type=float, default=10,
                         help='Sparsity constraint (default is 10\%)')
    rl_args.add_argument('--rl-size-constraint', type=float, default=10,
                         help='Memory size constraint (default is 10\%)')

    rl_agent_args = parser.add_argument_group("RL related arguments")
    rl_agent_args.add_argument('--rl-agent-verbose', action='store_true',
                                  help='Log various messages related to the RL agent')
    rl_agent_args.add_argument('--rl-agent-timesteps', '--rl-agent-total-timesteps',
                                  dest='rl_agent_total_timesteps', type=int, default=10000,
                                  help='Total timesteps for the agent learning (default: 10000)')
    rl_agent_args.add_argument('--rl-agent-train-episodes', '--rl-agent-episodes',
                                  type=int, dest='rl_agent_train_episodes',
                                  help="Number of episodes to train the agent. By default, timesteps are used as "
                                       "a timeout criterion, with the --rl-agent-timesteps command line argument")
    rl_agent_args.add_argument('--rl-agent-eval-episodes',
                                  type=int, default=10,
                                  help='Number of episodes to evaluate the learned policy. Default is 10 episodes')
    rl_agent_args.add_argument('--rl-agent-deterministic', '--rl-agent-det',
                                  action='store_true',
                                  help='Set for deterministic predictions from the agent')
    rl_agent_args.add_argument('--rl-agent-batch-size',
                                  type=int, default=64,
                                  help='Layer-level agent batch size. Default is 64')
    rl_agent_args.add_argument('--rl-agent-save-frequency',
                                  type=int, default=1,
                                  help='Frequency of saving the agent model (in timesteps). Default is 1')
    rl_agent_args.add_argument('--rl-agent-eval-frequency',
                                  type=int,
                                  help='Frequency of evaluating the agent model (in episodes).')
    rl_agent_args.add_argument('--rl-agent-no-improvement-evals', '--rl-agent-no-improv-evals',
                                  type=int, default=5, dest='rl_agent_no_improv_evals',
                                  help='Number of non-improving evaluations after which the training of the '
                                       'agent stops. Default is 5 evaluations')
    rl_agent_args.add_argument('--rl-agent-min-evals',
                                  type=int, default=10,
                                  help='Minimum number of evaluations of the agent policy to conduct before '
                                       'quiting after no improvement. Default is 10 evaluations')
    rl_agent_args.add_argument('--rl-agent-policy-device', '--rl-agent-device',
                                  dest='rl_agent_policy_device', choices=['cpu', 'cuda'], default='cpu',
                                  help='Device for running the agent policy. Default is CPU')
    rl_agent_args.add_argument('--rl-agent-load-from-path', '--rl-agent-load',
                                  dest='rl_agent_load_from_path',
                                  help='Specify path to load agent from')
    return parser


def accel_args(parser):
    """Arguments related to the hardware DNN accelerator
    """
    accel_args = parser.add_argument_group("Accelerator-related arguments")
    accel_args.add_argument('--accelerator-type', dest='accelerator_arch_type',
                            type=accelerator_type_arg, default='eyeriss',
                            help='Type of accelerator architecture. Default is Eyeriss-like')

    return parser


def check_args(args):
    """Check for logical errors in argument parsing
    """
    if args.resumed_checkpoint_path:
        assert len(args.arch) == len(args.resumed_checkpoint_path)
    assert 0 <= args.pruning_low <= args.pruning_high <= 1
    #assert 2 <= args.quant_low <= args.quant_high <= 8



### Enumerators for argument parsing ###

class OptimizerType(Enum):
    SGD = 1
    Adam = 2

def optimizer_type_arg(argstr):
    str_to_optimizer_type_map = {'sgd': OptimizerType.SGD,
                                 'adam': OptimizerType.Adam}
    try:
        return str_to_optimizer_type_map[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"--optimizer-type argument must be one of the following: "
                                         f"{str_to_optimizer_type_map.keys()}. Invalid argument {argstr}")


def pruning_group_type_arg(argstr):
    try:
        pruning_group_type_dict = {str(entry.name).lower(): entry for entry in PruningGroupType}
        return pruning_group_type_dict[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid argument {argstr} for --pruning-group-type argument")


def accelerator_type_arg(argstr):
    str_to_accelerator_type_map = {str(entry.name).lower(): entry for entry in AcceleratorType}
    try:
        return str_to_accelerator_type_map[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"--accelerator-type argument must be one of the following: "
                                         f"{str_to_accelerator_type_map.keys()}. Invalid argument {argstr}")


class ModelSummaryType(Enum):
    Compute = 1
    Sparsity = 2
    Dummy = 3

def model_summary_type_arg(argstr):
    str_to_summary_type_map = {'compute': ModelSummaryType.Compute,
                               'sparsity': ModelSummaryType.Sparsity,
                               '': ModelSummaryType.Dummy}
    try:
        return str_to_summary_type_map[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"--model-summary argument must be one of the following: "
                                         f"{str_to_summary_type_map.keys()}. Invalid argument {argstr}")


