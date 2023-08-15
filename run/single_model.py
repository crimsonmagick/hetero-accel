import logging
import traceback
import os.path
import wandb
import pickle
import torch
from copy import deepcopy
from types import SimpleNamespace
from collections import OrderedDict
from src.utils import env_cfg, hanlde_subapps
from src.net_wrapper import TorchNetworkWrapper

logger = logging.getLogger(__name__)


def model_setup(args):
    """Setup the model and handle sub-applications
    """
    # model-specific arguments for network wrapper
    model_args = SimpleNamespace(arch=args.arch,
                                 dataset_dir=args.datadir,
                                 dataset=args.dataset,
                                 batch_size=args.batch_size,
                                 workers=args.workers,
                                 validation_split=args.validation_split,
                                 effective_train_size=args.effective_train_size,
                                 effective_valid_size=args.effective_valid_size,
                                 effective_test_size=args.effective_test_size,
                                 gpus=args.gpus,
                                 cpu=args.cpu,
                                 load_serialized=args.load_serialized,
                                 pretrained=args.pretrained,
                                 resumed_checkpoint_path=args.resumed_checkpoint_path,
                                 optimizer_type=args.optimizer_type,
                                 learning_rate=args.learning_rate,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay,
                                 evaluate=args.evaluate_model_mode,
                                 profile_model=args.use_profiler,
                                 print_frequency=args.batch_print_frequency,
                                 save_frequency=args.model_save_frequency,
                                 verbose=args.model_verbose,
                                 logdir=args.logdir,
                                 )
    net_wrapper = TorchNetworkWrapper(model_args)

    wandb.init(project='hetero-accel')


    wandb.finish()


def main():
    """Main function for executing RL training or other operations
    """
    args = env_cfg()
    args.logdir = logging.getLogger().logdir

    # get model wrapper
    net_wrapper = model_setup(args)

    # execute other modes
    if handle_model_subapps(net_wrapper, args):
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

