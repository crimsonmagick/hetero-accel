import logging
import traceback
import os.path
import wandb
from copy import deepcopy
from collections import OrderedDict
from src import dataset_dirs, pretrained_checkpoint_paths
from src.utils import env_cfg, handle_model_subapps
from src.net_wrapper import dnn_setup
from src.dataset import load_data

logger = logging.getLogger(__name__)


def main():
    """Main function for executing RL training or other operations
    """
    #wandb.init(project='hetero-accel')

    args = env_cfg()
    args.logdir = logging.getLogger().logdir

    dnn_args = deepcopy(args)
    dnns = OrderedDict()
    datasets = OrderedDict()
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
        # load DNN wrapper
        net_wrapper = dnn_setup(dnn_args)
        dnns[arch] = net_wrapper

        do_exit = handle_model_subapps(net_wrapper, data_loaders, args)

    logger.info(f"{dnns}")
    logger.info(f"{datasets}")
    #wandb.finish()

    if do_exit:
        exit()


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

