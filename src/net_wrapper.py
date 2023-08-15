import torch
import logging
import os
import re
from copy import deepcopy
from collections import OrderedDict
from src.args import OptimizerType, AxLayerType
from src.utils import weight_init, load_checkpoint, model_io_summary, transform_model, save_checkpoint
from src.train_test import train, validate
from src.dataset import load_data
from src.models.torch import create_model

logger = logging.getLogger(__name__)


class TorchNetworkWrapper:
    def __init__(self, model_args):
        self.args = model_args
        self.config_compute_device()
        self.init_learner()
        self.init_data_loaders()

        # logging directory
        self.logdir = model_args.logdir

    def config_compute_device(self):
        if self.args.cpu or not torch.cuda.is_available():
            # Set GPU index to -1 if using CPU
            self.args.device = 'cpu'
            self.args.gpus = -1
        else:
            self.args.device = 'cuda'
            self.args.gpus = [self.args.gpus] if not isinstance(self.args.gpus, list) else self.args.gpus
            if self.args.gpus is not None:
                if len(self.args.gpus) == 1 and re.search('[ ,]', str(self.args.gpus[0])):
                    self.args.gpus = [int(device_idx.strip()) for device_idx in re.split('[ ,]', self.args.gpus[0])]
                else:
                    self.args.gpus = [int(device_idx) for device_idx in self.args.gpus]

                available_gpus = torch.cuda.device_count()
                for device_idx in self.args.gpus:
                    if device_idx >= available_gpus:
                        raise ValueError(f'ERROR: GPU device ID {device_idx} requested, but only {available_gpus} devices available')
                # Set default device in case the first one on the list != 0
                torch.cuda.set_device(self.args.gpus[0])

    def init_learner(self):
        # Create the model
        self.model = create_model(self.args.arch, self.args.dataset, self.args.pretrained,
                                  parallel=not self.args.load_serialized, device_ids=self.args.gpus,)
                                  #verbose=self.args.verbose)
        self.model.apply(weight_init)

        self.optimizer = None

        # load model from checkpoint
        if self.args.resumed_checkpoint_path is not None:
            self.model, self.optimizer, _ = load_checkpoint(
                self.model, self.args.resumed_checkpoint_path,
                model_device=self.args.device,
                to_cpu=self.args.device == 'cpu',
                #verbose=self.args.verbose
            )

        # save the loaded model
        self.original_model = deepcopy(self.model)

        # define optimizer
        if self.optimizer is None and not self.args.evaluate:
            self.init_optimizer()
            logger.debug(f'Optimizer Type: {type(self.optimizer)}')
            logger.debug(f'Optimizer args: {self.optimizer.defaults}')

        # define loss function (criterion)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def init_optimizer(self):
        """Initialize an optimizer instance"""
        if self.args.optimizer_type == OptimizerType.SGD:
            # SGD optimizer
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.args.learning_rate,
                                             momentum=self.args.momentum,
                                             weight_decay=self.args.weight_decay)
            # ADAM optimizer
        elif self.args.optimizer_type == OptimizerType.Adam:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.args.learning_rate,
                                              weight_decay=self.args.weight_decay)

    def init_data_loaders(self):
        """Load the dataset and data loaders
        """
        torch.cuda.empty_cache()
        self.train_loader, self.valid_loader, self.test_loader = load_data(
            self.args.dataset, os.path.expanduser(self.args.dataset_dir),
            self.model.arch,
            self.args.batch_size, self.args.workers,
            self.args.validation_split,
            self.args.effective_train_size,
            self.args.effective_valid_size,
            self.args.effective_test_size,
            self.args.evaluate,
            True, #self.args.verbose
        )


    def log_model(self, tb_logger):
        """Record the graph of the model and its various statistics using TensorBoard
        """
        if self.args.evaluate:
            return

        # save the model graph using Tensorboard
        inputs, labels = next(iter(self.test_loader))
        tb_logger.add_graph(self.model, inputs)

        # profile the model operations and GPU utilization
        if self.args.profile_model:
            profiler = torch.profiler.profile(
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(self.logdir),
                        record_shapes=True,
                        with_stach=True)
            self.train(epochs=1,
                       steps_per_epoch=(1 + 1 + 3) * 2,
                       profiler=profiler)

        # histograms of model parameters and their gradients
        for name, param in self.model.named_parameters():
            tb_logger.add_histogram(name, param, 0)
            if param.grad is not None:
                tb_logger.add_histogram(name + '.grad', param.grad, 0)

    def train(self, epochs, steps_per_epoch=None, profiler=None):
        """Run some training epochs on the model
        """
        train_metrics = []
        if steps_per_epoch is None:
            steps_per_epoch = len(self.train_loader)

        for epoch in range(epochs):
            top1, top5, loss = train(self.train_loader, self.model, self.criterion, self.optimizer, profiler,
                                     None, epoch, steps_per_epoch, self.args.verbose, self.args.print_frequency)
            train_metrics.extend((top1, top5, loss))
        return train_metrics

    def validate(self):
        """Run inference on the validation set
        """
        logger.debug(f"Running inference on validation set. Model outline:\n{self.model}")
        top1, top5, loss = validate(self.valid_loader, self.model, self.criterion, 0,
                                    self.args.verbose, self.args.print_frequency)
        return top1, top5, loss

    def test(self):
        """Run inference on the test set
        """
        logger.debug(f"Running inference on test set. Model outline:\n{self.model}")
        top1, top5, loss = validate(self.test_loader, self.model, self.criterion, 0,
                                    self.args.verbose, self.args.print_frequency)
        return top1, top5, loss


    def save_model(self, episode, is_best):
        """Save the current version of the model, only if the episode is the best so far,
           or the saving frequency is set accordingly
        """
        if is_best or (getattr(self.args, 'save_frequency', None) and 
                       episode + 1 % self.args.save_frequency == 0):

            save_checkpoint(arch=self.model.arch, model=self.model,
                            epoch=episode, is_best=is_best, savedir=self.logdir, verbose=self.args.verbose)
            logger.debug(f"Saved model:\n{self.model}")

