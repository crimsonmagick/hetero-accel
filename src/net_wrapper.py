import torch
import logging
import os
import re
from types import SimpleNamespace
from copy import deepcopy
from collections import OrderedDict
from src import pretrained_checkpoint_paths
from src.utils import weight_init, load_checkpoint, model_summary, transform_model, save_checkpoint
from src.train_test import train, validate
from src.dataset import load_data
from src.models import create_model
from src.args import OptimizerType

logger = logging.getLogger(__name__)


class TorchNetworkWrapper:
    """DNN wrapper with training/testing functionality
    """
    def __init__(self, args, model=None):
        for name, value in vars(args).items():
            setattr(self, name, value)

        self.config_compute_device()
        self.model = model
        if model is None:
            self.init_model()
        self.summary = model_summary(self.model)
        self.num_layers = len(self.summary)

    @classmethod
    def from_args(cls, args):
        args = SimpleNamespace(arch=args.arch,
                               dataset=args.dataset,
                               batch_size=args.batch_size,
                               gpus=args.gpus,
                               cpu=args.cpu,
                               load_serialized=args.load_serialized,
                               pretrained=args.pretrained,
                               resumed_checkpoint_path=args.resumed_checkpoint_path,
                               print_frequency=args.batch_print_frequency,
                               verbose=args.model_verbose,
                               logdir=args.logdir,
                               )
        return cls(args)
         
    def config_compute_device(self):
        if self.cpu or not torch.cuda.is_available():
            # Set GPU index to -1 if using CPU
            self.device = 'cpu'
            self.gpus = -1
        else:
            self.device = 'cuda'
            self.gpus = [self.gpus] if not isinstance(self.gpus, list) else self.gpus
            if self.gpus is not None:
                if len(self.gpus) == 1 and re.search('[ ,]', str(self.gpus[0])):
                    self.gpus = [int(device_idx.strip()) for device_idx in re.split('[ ,]', self.gpus[0])]
                else:
                    self.gpus = [int(device_idx) for device_idx in self.gpus]

                available_gpus = torch.cuda.device_count()
                for device_idx in self.gpus:
                    if device_idx >= available_gpus:
                        raise ValueError(f'ERROR: GPU device ID {device_idx} requested, but only {available_gpus} devices available')
                # Set default device in case the first one on the list != 0
                torch.cuda.set_device(self.gpus[0])

    def init_model(self):
        """Initialize the DNN architecture with the option of pretrained weights
        """
        assert not (self.pretrained and self.resumed_checkpoint_path is not None)
        if self.pretrained and self.arch + '_' + self.dataset in pretrained_checkpoint_paths:
            self.pretrained = False
            self.resumed_checkpoint_path = pretrained_checkpoint_paths[self.arch + '_' + self.dataset]

        self.model = create_model(self.arch,
                                  self.dataset,
                                  self.batch_size,
                                  self.pretrained,
                                  parallel=not self.load_serialized,
                                  device_ids=self.gpus,)
                                  #verbose=self.verbose)
        self.model.apply(weight_init)

        if self.resumed_checkpoint_path is not None:
            self.model, _ = load_checkpoint(
                self.model,
                self.resumed_checkpoint_path,
                model_device=self.device,
                to_cpu=self.device == 'cpu',
                #verbose=self.verbose
            )

    def log_model(self, test_loader, tb_logger):
        """Record the graph of the model and its various statistics using TensorBoard
        """
        # save the model graph using Tensorboard
        inputs, labels = next(iter(test_loader))
        tb_logger.add_graph(self.model, inputs)

        # histograms of model parameters and their gradients
        for name, param in self.model.named_parameters():
            tb_logger.add_histogram(name, param, 0)
            if param.grad is not None:
                tb_logger.add_histogram(name + '.grad', param.grad, 0)

    def train(self, epochs, train_loader, criterion, optimizer, steps_per_epoch=None, profiler=None):
        """Run some training epochs on the model
        """
        train_metrics = []
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)

        for epoch in range(epochs):
            top1, top5, loss = train(train_loader, self.model, criterion, optimizer, profiler,
                                     None, epoch, steps_per_epoch,
                                     self.verbose, self.print_frequency)
            train_metrics.extend((top1, top5, loss))
        return train_metrics

    def validate(self, valid_loader, criterion):
        """Run inference on the validation set
        """
        logger.debug(f"Running inference on validation set")
        top1, top5, loss = validate(valid_loader, self.model, criterion, 0,
                                    self.verbose, self.print_frequency)
        return top1, top5, loss

    def test(self, test_loader, criterion):
        """Run inference on the test set
        """
        logger.debug(f"Running inference on test set")
        top1, top5, loss = validate(test_loader, self.model, criterion, 0,
                                    self.verbose, self.print_frequency)
        return top1, top5, loss


    def save_model(self, name=None, episode=None, is_best=False, verbose=True):
        """Save the current version of the model
        """
        save_checkpoint(arch=self.model.arch, model=self.model,
                        epoch=episode, is_best=is_best, verbose=verbose,
                        name=name, savedir=self.logdir)

