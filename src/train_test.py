import logging
import time
import math
from contextlib import nullcontext
from brevitas_examples.imagenet_classification.utils import validate as validate_quant


import brevitas_examples
import torch
import torchnet.meter as tnt
from collections import OrderedDict

from brevitas.export.inference import quant_inference_mode

from src.utils import log_training_progress


logger = logging.getLogger(__name__)


def train(train_loader, model, criterion, optimizer, profiler, accuracy_meter,
          compression_scheduler, epoch, steps_per_epoch, verbose, print_frequency):
    """Training-with-compression loop for one epoch"""
    def _log_training_progress():
        stats_dict = OrderedDict()
        for accuracy_metric in accuracy_meter.metrics:
            stats_dict[accuracy_metric] = accuracy_meter.value(metric=accuracy_metric)
        for loss_name, meter in losses.items():
            stats_dict[loss_name] = meter.mean
        stats_dict['LR'] = optimizer.param_groups[0]['lr']
        stats_dict['Time'] = batch_time.mean
        log_training_progress(stats_dict, epoch, steps_completed, steps_per_epoch)

    losses = OrderedDict([('Overall Loss', tnt.AverageValueMeter()),
                          ('Objective Loss', tnt.AverageValueMeter())])

    device = model.device
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)

    if compression_scheduler is not None:
        compression_scheduler.on_epoch_begin(model, epoch)

    if profiler is not None:
        profiler.start()

    # Switch to train mode
    model.train()

    if verbose:
        logger.info(f'Training epoch {epoch}: {total_samples} samples ({batch_size} per mini-batch)')
    model.to(device)
    end = time.time()

    for train_step, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to(device), target.to(device)

        if compression_scheduler is not None:
            compression_scheduler.before_forward_pass(model)

        output = model(inputs)
        loss = criterion(output, target)

        # record loss
        losses['Objective Loss'].add(loss.item())

        # measure accuracy
        if isinstance(output, tuple):
            accuracy_meter.add(output[0].detach(), target)
        else:
            accuracy_meter.add(output.detach(), target)

        # aggregate loss from regularization terms
        if compression_scheduler is not None:
            agg_loss = compression_scheduler.before_backward_pass(model, loss)
            if agg_loss is not None:
                loss = agg_loss.overall_loss
                losses['Overall Loss'].add(loss.item())

                for lc in agg_loss.loss_components:
                    if lc.name not in losses:
                        losses[lc.name] = tnt.AverageValueMeter()
                    losses[lc.name].add(lc.value.item())

        # compute the gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if compression_scheduler is not None:
            compression_scheduler.before_parameter_optimization(model)

        optimizer.step()
        if compression_scheduler is not None:
            compression_scheduler.on_minibatch_end(model, is_last_minibatch=train_step == (steps_per_epoch - 1))

        if profiler is not None:
            profiler.step()

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()
        steps_completed = train_step + 1

        if steps_completed % print_frequency == 0:
            _log_training_progress()

    if compression_scheduler is not None:
        compression_scheduler.on_epoch_end(epoch)

    if profiler is not None:
        profiler.stop()

    return *accuracy_meter.value(metric='all'), losses['Overall Loss'].mean


def validate(valid_loader, model, criterion, accuracy_meter, epoch, verbose, print_frequency, use_quant=False):
    if verbose:
        logger.info(f'--- validate (epoch={epoch})-----------')

    def _log_validation_progress():
        stats_dict = OrderedDict()
        for accuracy_metric in accuracy_meter.metrics:
            stats_dict[accuracy_metric] = accuracy_meter.value(metric=accuracy_metric)
        stats_dict['Loss'] = losses['Objective Loss'].mean
        stats_dict['Time'] = batch_time.mean
        log_training_progress(stats_dict, epoch, steps_completed, total_steps)

    """Execute the validation/test loop."""
    losses = {'Objective Loss': tnt.AverageValueMeter()}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_time = tnt.AverageValueMeter()
    total_samples = len(valid_loader.sampler)
    batch_size = valid_loader.batch_size
    total_steps = total_samples / batch_size
    if verbose:
        logger.info(f'{total_samples} samples ({batch_size} per mini-batch)')

    # Switch to evaluation mode
    model.eval()
    model.to(device)

    end = time.time()
    quant_cm = quant_inference_mode(model) if use_quant else nullcontext()
    with torch.no_grad(), quant_cm:
        # if use_quant:
        #     quant_top1 = validate_quant(valid_loader, model, stable=True)
        #     print("QUANT_TOP1: " + quant_top1)
        for validation_step, (inputs, target) in enumerate(valid_loader):

            # cast to device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
            if isinstance(target, torch.Tensor):
                target = target.to(device)

            # compute output from model
            output = model(inputs)

            # compute loss
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses['Objective Loss'].add(loss.item())
            accuracy_meter.add(output, target)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = validation_step + 1
            if verbose and steps_completed % print_frequency == 0:
                _log_validation_progress()

    if verbose:
        logstr = '   '.join([
            f'{accuracy_metric.capitalize()}: {accuracy_meter.value(metric=accuracy_metric):.3f}'
            for accuracy_metric in accuracy_meter.metrics
        ])
        logger.info(f"==> {logstr}   Loss: {losses['Objective Loss'].mean:.3f}\n")

    return *accuracy_meter.value(metric='all'), losses['Objective Loss'].mean

