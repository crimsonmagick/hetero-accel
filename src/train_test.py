import logging
import time
import math
import torch
import torch.nn.functional as F
import torchnet.meter as tnt
from einops import rearrange
from collections import OrderedDict
from src.utils import log_training_progress 

logger = logging.getLogger(__name__)


def train(train_loader, model, criterion, optimizer, profiler,
          compression_scheduler, epoch, steps_per_epoch, verbose, print_frequency):
    """Training-with-compression loop for one epoch"""
    def _log_training_progress():
        stats_dict = OrderedDict()
        stats_dict['Top1'] = classerr.value(1)
        stats_dict['Top5'] = classerr.value(5)
        for loss_name, meter in losses.items():
            stats_dict[loss_name] = meter.mean
        stats_dict['LR'] = optimizer.param_groups[0]['lr']
        stats_dict['Time'] = batch_time.mean

        log_training_progress(stats_dict, epoch, steps_completed, steps_per_epoch)

    losses = OrderedDict([('Overall Loss', tnt.AverageValueMeter()),
                          ('Objective Loss', tnt.AverageValueMeter())])

    device = model.device
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
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
    acc_stats = []
    end = time.time()

    for train_step, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to(device), target.to(device)

        if compression_scheduler is not None:
            compression_scheduler.before_forward_pass(model)

        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy
        if isinstance(output, tuple):
            classerr.add(output[0].detach(), target)
        else:
            classerr.add(output.detach(), target)
        acc_stats.append([classerr.value(1), classerr.value(5)])

        # record loss
        losses['Objective Loss'].add(loss.item())

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
        steps_completed = train_step + 1

        if steps_completed % print_frequency == 0:
            _log_training_progress()

        end = time.time()

    if compression_scheduler is not None:
        compression_scheduler.on_epoch_end(epoch)

    if profiler is not None:
        profiler.stop()

    return classerr.value(1), classerr.value(5), losses['Overall Loss']


def validate(valid_loader, model, criterion, epoch, verbose, print_frequency):
    if verbose:
        logger.info(f'--- validate (epoch={epoch})-----------')

    def _log_validation_progress():
        stats_dict = OrderedDict([('Loss', losses['objective_loss'].mean),
                                  ('Top1', classerr.value(1)),
                                  ('Top5', classerr.value(5))])
        log_training_progress(stats_dict, epoch, steps_completed, total_steps)

    """Execute the validation/test loop."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

    device = model.device
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
    with torch.no_grad():
        for validation_step, (inputs, target) in enumerate(valid_loader):
            inputs, target = inputs.to(device), target.to(device)
            # compute output from model
            output = model(inputs)

            # compute loss
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses['objective_loss'].add(loss.item())
            classerr.add(output.detach(), target)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = validation_step + 1
            if verbose and steps_completed % print_frequency == 0:
                _log_validation_progress()

    if verbose:
        logger.info('==> Top1: {:.3f}    Top5: {:.3f}    Loss: {:.3f}\n'.format(
                       classerr.value(1), classerr.value(5), losses['objective_loss'].mean))
    return classerr.value(1), classerr.value(5), losses['objective_loss'].mean

