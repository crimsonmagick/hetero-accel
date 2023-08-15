import os
from enum import Enum
from src.models import ALL_MODELS
from src.dataset import ALL_DATASETS


def app_args(parser):
    """Arguments for running the main application"""
    parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs',
                         help='Path to dump logs and checkpoints')
    parser.add_argument('--name', '-n', help='Experiment name')
    parser.add_argument('--deterministic', '--det', action='store_true',
                        help='Set for deterministic execution. Preferably, set the experiment --seed as well')
    parser.add_argument('--seed', dest='global_seed', type=int,
                        help='Global seed for deterministic execution')
    parser.add_argument('--yaml-cfg-file', default=os.path.join(project_dir, 'run', 'args_cfg.yaml'), 
                        help='YAML file containing the experiment description')
    parser.add_argument('--load-layer-lut-from', help='Path to .pkl file whith results from the first experiment (layer-level)')

    op_mode = parser.add_argument_group("Execution mode arguments")
    op_mode = op_mode.add_mutually_exclusive_group()
    op_mode.add_argument('--evaluate-model', dest='evaluate_model_mode', action='store_true',
                         help='Evaluate DNN model on test set')
    op_mode.add_argument('--train-model', dest='train_model_mode', action='store_true',
                         help='Train DNN model on train set')
 
    return parser


def model_args(parser):
    """Arguments for configuring the application model"""
    model_args = parser.add_argument_group("DNN related arguments")
    model_args.add_argument('-a', '--arch', choices=ALL_MODELS,
                            help='Define NN model. Options: {}'.format('|'.join(ALL_MODELS)))
    model_args.add_argument('--datadir', metavar='DATASET_DIR', help='Path to dataset.')
    model_args.add_argument('--dataset', '-d', choices=ALL_DATASETS, default='cifar10',
                            help=f"Select dataset. Options: {'|'.join(ALL_DATASETS)}")
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
    model_args.add_argument('--retrain-epochs', type=int, default=0,
                            help='Number of retraining epochs after episode completion. Disabled by default.')

    load_checkpoint_group = parser.add_argument_group('Resuming arguments')
    load_checkpoint_group_exc = load_checkpoint_group.add_mutually_exclusive_group()
    load_checkpoint_group_exc.add_argument('--resume-from', dest='resumed_checkpoint_path', metavar='PATH',
                                           help='path to latest checkpoint. Use to resume paused training session.')
    load_checkpoint_group_exc.add_argument('--pretrained', action='store_true', help='Use pre-trained model')

    optimizer_args = parser.add_argument_group('Optimizer arguments')
    optimizer_args.add_argument('--optimizer-type', '--ot', type=optimizer_type_arg, default='sgd',
                                help='Choose optimizer type. Choices: ' + ' | '.join(str_to_optimizer_type_map.keys()))
    optimizer_args.add_argument('--learning-rate', '--lr', default=0.1, type=float, help='Initial learning rate (default: 0.1)')
    optimizer_args.add_argument('--momentum', default=0.9, type=float, help='Momentum (default: 0.9)')
    optimizer_args.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='Weight decay (default: 1e-4)')

    return parser



