import torch
import logging
import os
import re
from types import SimpleNamespace
from copy import deepcopy
from collections import OrderedDict
from src.utils import weight_init, load_checkpoint, model_summary, transform_model, save_checkpoint
from src.train_test import train, validate
from src.dataset import load_data
from src.models import create_model
from src.args import OptimizerType

logger = logging.getLogger(__name__)


class TorchNetworkWrapper:
    """DNN wrapper with training/testing functionality
    """
    def __init__(self, model_args):
        self.args = model_args
        self.config_compute_device()
        self.init_model()
        self.logdir = self.args.logdir
        self.summary = model_summary(self.model)
        self.num_layers = len(self.summary)

    @classmethod
    def from_args(cls, args):
        model_args = SimpleNamespace(arch=args.arch,
                                     dataset=args.dataset,
                                     gpus=args.gpus,
                                     cpu=args.cpu,
                                     load_serialized=args.load_serialized,
                                     pretrained=args.pretrained,
                                     resumed_checkpoint_path=args.resumed_checkpoint_path,
                                     profile_model=args.use_profiler,
                                     print_frequency=args.print_frequency,
                                     verbose=args.verbose,
                                     logdir=args.logdir,
                                     )
        return cls(model_args)
         
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

    def init_model(self):
        self.model = create_model(self.args.arch,
                                  self.args.dataset,
                                  self.args.pretrained,
                                  parallel=not self.args.load_serialized,
                                  device_ids=self.args.gpus,)
                                  #verbose=self.args.verbose)
        self.model.apply(weight_init)

        if self.args.resumed_checkpoint_path is not None:
            self.model, _ = load_checkpoint(
                self.model,
                self.args.resumed_checkpoint_path,
                model_device=self.args.device,
                to_cpu=self.args.device == 'cpu',
                #verbose=self.args.verbose
            )

    def log_model(self, test_loader, tb_logger):
        """Record the graph of the model and its various statistics using TensorBoard
        """
        # save the model graph using Tensorboard
        inputs, labels = next(iter(test_loader))
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

    def train(self, epochs, train_loader, criterion, optimizer, steps_per_epoch=None, profiler=None):
        """Run some training epochs on the model
        """
        train_metrics = []
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)

        for epoch in range(epochs):
            top1, top5, loss = train(train_loader, self.model, criterion, optimizer, profiler,
                                     None, epoch, steps_per_epoch,
                                     self.args.verbose, self.args.print_frequency)
            train_metrics.extend((top1, top5, loss))
        return train_metrics

    def validate(self, valid_loader, criterion):
        """Run inference on the validation set
        """
        logger.debug(f"Running inference on validation set. Model outline:\n{self.model}")
        top1, top5, loss = validate(valid_loader, self.model, criterion, 0,
                                    self.args.verbose, self.args.print_frequency)
        return top1, top5, loss

    def test(self, test_loader, criterion):
        """Run inference on the test set
        """
        logger.debug(f"Running inference on test set. Model outline:\n{self.model}")
        top1, top5, loss = validate(test_loader, self.model, criterion, 0,
                                    self.args.verbose, self.args.print_frequency)
        return top1, top5, loss


    def save_model(self, episode, is_best, verbose=True):
        """Save the current version of the model
        """
        save_checkpoint(arch=self.model.arch, model=self.model,
                        epoch=episode, is_best=is_best, savedir=self.logdir, verbose=verbose)
        logger.debug(f"Saved model:\n{self.model}")

