import logging
import traceback
import os.path
import pickle
import numpy as np
import pandas as pd
import yaml
from copy import deepcopy
from types import SimpleNamespace
from src import dataset_dirs
from src.workload import MultiDNNWorkload
from src.utils import env_cfg, handle_model_subapps
from src.args import OperationMode
from src.net_wrapper import TorchNetworkWrapper
from src.compression.compressor import PruningQuantizationCompressor
from src.dataset import load_data
from src.accelerator_cfg import AcceleratorProfile
from src.optimizer import AcceleratorOptimizer
from src.baseline import run_baseline
from src.sota import run_sota


logger = logging.getLogger(__name__)


def main():
    """Main executing function, supporting the execution of either
       our optimization, or others for comparisons
    """
    args = env_cfg()
    args.logdir = logging.getLogger().logdir
    # save arguments as pkl, for reproducibility
    with open(os.path.join(args.logdir, 'args.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)

    # initialize the workload
    workload = setup_workload(args)
    # create a LUT of quantization profiles for each DNN-precision pairing
    dnn_accuracy_lut, compressors = quant_exploration(args, workload)

    if args.operation_mode == OperationMode.Ours:
        # perform a DSE to define the sub-accelerator architectures
        accel, optimizer = accelerator_exploration(args, workload, dnn_accuracy_lut)
        # prune the DNNs to produce the final scheduler
        schedule, metrics = pruned_schedule(args, workload, dnn_accuracy_lut,
                                            compressors, optimizer, accel)

    # evaluate a given baseline accelerator architecture
    elif args.operation_mode == OperationMode.Baseline:
        run_baseline(args, workload, dnn_accuracy_lut)

    # execute the optimizations in the state-of-the-art
    elif args.operation_mode == OperationMode.SOTA:
        run_sota(args, workload, dnn_accuracy_lut)


def setup_workload(args):
    """Initialize the multi-DNN workload
    """
    dnn_args = deepcopy(args)
    dnns = {}
    datasets = {}
    print_frequency = {}

    # load workload file
    with open(args.workload_cfg_file, 'r') as stream:
        workloads = yaml.safe_load(stream)
    multi_dnn_workload = workloads['workloads'][args.workload_idx]

    # setup each DNN separately
    for idx, workload_dict in enumerate(multi_dnn_workload):

        # gather workload arguments
        for name, value in workload_dict.items():
            setattr(dnn_args, name, value)
        # specific handling of batch size
        if 'batch_size' not in workload_dict:
            dnn_args.batch_size = args.batch_size

        # configure and save DNN wrapper
        net_wrapper = TorchNetworkWrapper.from_args(dnn_args)
        dnns[net_wrapper.model.arch] = net_wrapper
        print_frequency[net_wrapper.model.arch] = dnn_args.batch_print_frequency

        if dnn_args.dataset not in datasets:
            # configure dataset
            data_loaders = load_data(
                dnn_args.dataset,
                dataset_dirs[dnn_args.dataset],
                net_wrapper.model.arch,
                dnn_args.batch_size,
                args.workers,
                args.validation_split,
                args.effective_train_size,
                args.effective_valid_size,
                args.effective_test_size,
                args.evaluate_model_mode,
                True, #self.args.verbose
            )
            datasets[dnn_args.dataset] = data_loaders

        # manually execute the summary, if not already done
        if getattr(net_wrapper, 'summary', None) is None:
            net_wrapper.run_summary(datasets[dnn_args.dataset][0])

        # execute sub-applications
        if handle_model_subapps(net_wrapper, data_loaders, args):
            exit(0)

    return MultiDNNWorkload(dnns, datasets, print_frequency)


def init_compressor(args, workload, arch, net_wrapper):
    compression_args = SimpleNamespace(logdir=args.logdir,
                                       pruning_high=args.pruning_high,
                                       pruning_low=args.pruning_low,
                                       quant_high=args.quant_high,
                                       quant_low=args.quant_low,
                                       layer_type_whitelist=args.layer_type_whitelist,
                                       pruning_group_type=args.pruning_group_type,
                                       accelerator_cfg=AcceleratorProfile(args.accelerator_arch_type),
                                       # DNN args for inheritance from TorchNetworkWrapper
                                       optimizer_type=args.optimizer_type,
                                       profile_model=False,
                                       gpus=args.gpus,
                                       cpu=args.cpu,
                                       print_frequency=workload.print_frequency[arch],
                                       verbose=args.model_verbose)
    return PruningQuantizationCompressor(compression_args,
                                         workload.datasets[net_wrapper.model.dataset],
                                         net_wrapper.model)


def quant_exploration(args, workload):
    """Exploration of possible quantization profiles
    """
    skip_exploration = False
    if getattr(args, 'dnn_accuracy_lut_file', None) is not None and \
       os.path.exists(args.dnn_accuracy_lut_file):
        skip_exploration = True
        preloaded_dnn_accuracy_lut = pd.read_csv(args.dnn_accuracy_lut_file)
        assert all(arch in preloaded_dnn_accuracy_lut['Network'].unique() for arch in workload.dnns), \
            f"All DNNs {list(workload.dnns.keys())} must be included in the preloaded accuracy LUT: {args.dnn_accuracy_lut_file}"
        logger.info(f'=> Skipping exhaustive exploration: loaded LUT from {args.dnn_accuracy_lut_file}')

    # structure of the LUT
    df = pd.DataFrame(columns=['Network', 'QuantBits', 'Accuracy', 'Sparsity', 'Size', 'Valid'])

    compressors = {}
    for arch, net_wrapper in workload.dnns.items():
        if skip_exploration:
            continue

        # check if there is at least one accuracy constraint set
        assert any(
            getattr(args, f'{metric}_constraint', None) is not None for metric in net_wrapper.accuracy_metrics
        ), f"No accuracy constraint was set! Define at least one of the following in {args.yaml_cfg_file}: " \
           f"{' | '.join([f'{metric}_constraint'] for metric in net_wrapper.accuracy_metrics)}"

        # initialize compressor
        compressor = init_compressor(args, workload, arch, net_wrapper)
        compressors[arch] = compressor
        logger.info(f'=> Beginning exhaustive exploration for {arch}')

        # compute original statistics
        accuracy_stats = compressor.validate() if args.use_validation_set else compressor.test()
        model_stats, _ = compressor.compute_model_statistics()
        og_stats = {
            'accuracy': accuracy_stats[0], 
            'sparsity': model_stats['sparsity'], 'size': model_stats['size'],
            **{
                metric: accuracy_stats[i] for i, metric in enumerate(net_wrapper.accuracy_metrics)
            }
        }
        og_stats_logstr = ', '.join([f'{metric.capitalize()}={value:.2f}' if metric != 'size' else
                                     f'{metric.capitalize()}={value:.2e}'
                                     for metric, value in og_stats.items()])
        logger.info(f'{arch}: Original statistics: {og_stats_logstr}')
        # save the statistics to the LUT
        df.loc[len(df.index)] = ([arch, 32, accuracy_stats[0], model_stats['sparsity'], model_stats['size'], 1])

        # iterate over quantization bits
        quant_bits_options = np.arange(args.quant_low, args.quant_high + 1, args.quant_incr)
        if 16 not in quant_bits_options:
            quant_bits_options = np.append(quant_bits_options, 16)

        for quant_bits in quant_bits_options:
            logger.info(f'{arch}: Testing quantization of {quant_bits} bits')

            # reset the previous state of the network
            compressor.reset()
            # execute the compression profile
            compressor.quantize(quant_bits)
            # evaluate for accuracy and network statistics
            accuracy_stats = compressor.validate() if args.use_validation_set else compressor.test()
            model_stats, _ = compressor.compute_model_statistics()
            stats = {
                'accuracy': accuracy_stats[0],
                'sparsity': model_stats['sparsity'], 'size': model_stats['size'],
                **{
                    metric: accuracy_stats[i] for i, metric in enumerate(net_wrapper.accuracy_metrics)
                }
            }
            stats_logstr = ', '.join([f'{metric.capitalize()}={value:.2f}' if metric != 'size' else
                                      f'{metric.capitalize()}={value:.2e}'
                                      for metric, value in stats.items()])
            logger.info(f'{arch}: Compressed statistics: {stats_logstr}')

            # binary flag whether at least one of the accuracy constraints are satisfied
            valid = 0
            if all(
                getattr(args, f'{metric}_constraint', None) is None or
                accuracy_stats[i] >= og_stats[metric] - getattr(args, f'{metric}_constraint')
                for i, metric in enumerate(net_wrapper.accuracy_metrics)
            ):
                valid = 1

            # save the statistics to the LUT
            df.loc[len(df.index)] = ([arch, quant_bits, accuracy_stats[0], model_stats['sparsity'], model_stats['size'], valid])

    if skip_exploration:
       df = preloaded_dnn_accuracy_lut

    # check if any valid solutions were found
    assert df['Valid'].sum() > 1, "No valid solutions were found, consider changing the compression settings or " \
                                  "loosen the accuracy constraints"

    # save LUT to .csv file
    df.to_csv(os.path.join(args.logdir, 'lut.csv'))

    return df, compressors


def accelerator_exploration(args, workload, accuracy_lut):
    """Exploration to design/discover the sub-accelerator architectures
    """
    precision_options = sorted(set(
        accuracy_lut.loc[accuracy_lut['Valid'] == 1]['QuantBits']
    ))
    # remove 16 and 32 bits from the options
    try:
        precision_options.remove(32)
    except ValueError:
        pass
    try:
        precision_options.remove(16)
    except ValueError:
        pass

    # check if there is at least one valid option for each DNN
    if any(
        accuracy_lut.loc[
            (accuracy_lut['Network'] == arch) & 
            (accuracy_lut['QuantBits'] != 32) & 
            (accuracy_lut['QuantBits'] != 16)
        ]['Valid'].sum() == 0
        for arch in workload.dnns
    ):
        which_dnns = [
            arch for arch in workload.dnns
            if accuracy_lut.loc[
                (accuracy_lut['Network'] == arch) & 
                (accuracy_lut['QuantBits'] != 32) & 
                (accuracy_lut['QuantBits'] != 16)
            ]['Valid'].sum() == 0
        ]
        raise ValueError("The following DNNs cannot be used, as all quantization options "
                         f"violate the accuracy constraint: {', '.join(which_dnns)}")

    accel_cfg = AcceleratorProfile(args.accelerator_arch_type)
    accel_cfg.design_space['precision'] = precision_options

    logger.debug(f"Examining design space: {accel_cfg.design_space}")

    # initalize and run optimizer
    optimizer = AcceleratorOptimizer(args=args,
                                     num_accelerators=len(precision_options),
                                     accelerator_cfg=accel_cfg,
                                     workload=workload,
                                     accuracy_lut=accuracy_lut,
                                     hw_constraints=SimpleNamespace(deadline=args.deadline_constraint,
                                                                    area=args.area_constraint)
                                     )

    if not args.skip_exploration:
        optimizer.run()

    logger.info("*------------------*")
    comment = ''
    if args.skip_exploration and getattr(args, 'load_state_from', None) is not None:
        comment = f' (loaded from: {args.load_state_from}) ' 
    logger.info(f"Final heterogeneous accelerator{comment}:")
    for state in optimizer.best_state:
        logger.info(f'\t{state}')
    logger.info("*------------------*")

    return optimizer.best_state, optimizer


def pruned_schedule(args, workload, dnn_accuracy_lut, compressors, optimizer, hetero_accel):
    """Prune the DNNs according to the final heterogeneous accelerator architecture
    """
    def lut_entry(arch, precision):
        return dnn_accuracy_lut.loc[
                (dnn_accuracy_lut['Network'] == arch) &
                (dnn_accuracy_lut['QuantBits'] == precision)
               ]

    def is_valid(arch, accuracy_metric, accuracy_value):
        if isinstance(accuracy_value, pd.Series):
            accuracy_value = accuracy_value.iloc[0]

        og_metric = lut_entry(arch, 32)['Accuracy'].iloc[0]
        accuracy_metric = accuracy_metric.lower()
        assert getattr(args, accuracy_metric + '_constraint', None) is not None
        return accuracy_value >= og_metric - getattr(args, accuracy_metric + '_constraint')

    # load measurements
    skip_mapping = False
    if getattr(args, 'load_pruned_mappings_from', None) is not None and \
       os.path.exists(args.load_pruned_mappings_from):
        skip_mapping = True
        with open(args.load_pruned_mappings_from, 'rb') as f:
            pruned_mappings = pickle.load(f)
        logger.info(f"=> Loaded pruned mappings from {args.load_pruned_mappings_from}")

    else:
        # create dictionaries to save measurements
        edp_dict = {key: optimizer.energy_dict[key] * optimizer.latency_dict[key]
                    for key in optimizer.energy_dict}
        pruned_mappings = SimpleNamespace(energy=deepcopy(optimizer.energy_dict),
                                            latency=deepcopy(optimizer.latency_dict),
                                            edp=edp_dict,
                                            area=deepcopy(optimizer.area_dict))

    # re-initialize compressors if they were not loaded
    if len(compressors) == 0:
        for arch, net_wrapper in workload.dnns.items():
            compressors[arch] = init_compressor(args, workload, arch, net_wrapper)

    logger.info("=> Beginning accelerator-aware pruning")

    # iterate over each accelerator
    for accelerator in hetero_accel:
        if skip_mapping:
            continue

        logger.info(f"\tEvaluating accelerator: {accelerator}")
        # iterate over each DNN
        for arch, compressor in compressors.items():
            logger.info(f"\t\tCompressing DNN: {arch}")

            # check the accuracy of the quantized DNN
            entry = lut_entry(arch, accelerator.precision)
            accuracy_satisfied = entry['Valid'].iloc[0]
            if not accuracy_satisfied:
                logger.info(f"\t\tSkipping pruning of quantized DNN: accuracy already violated")
                # Invalid scheduling mappings are marked with negative weight (latency)
                pruned_mappings.energy[(arch, accelerator)] = -1
                pruned_mappings.latency[(arch, accelerator)] = -1
                pruned_mappings.edp[(arch, accelerator)] = -1
                continue

            accuracy_metric = compressor.accuracy_metrics[0]

            # check compliance with accuracy threshold, in case of mismatches from previous run
            if accuracy_satisfied and not is_valid(arch, accuracy_metric, entry['Accuracy']):
                logger.warning(f"A solution was accepted but surpasses the given constraint")

            # increamentally prune the DNN until an accuracy violation
            prune_ratio = 0.0
            accuracy = entry['Accuracy']
            while accuracy_satisfied:
                prune_ratio += 0.05

                # execute quantization and pruning with new ratio
                compressor.reset()
                compressor.prune_and_quantize(prune_ratio, accelerator.precision)

                # evaluate for accuracy and network statistics
                accuracy, *_ = compressor.validate() if args.use_validation_set else compressor.test()
                # check constraint again
                accuracy_satisfied = is_valid(arch, accuracy_metric, accuracy)

            # continue from lastly pruned model that satisfied the accuracy constraint
            prune_ratio = max(0.0, prune_ratio - 0.05)

            # if no pruning can be handled, then skip the evaluation, as the metrics from the
            #  exact DNN-accelerator mapping are already stored
            if prune_ratio == 0.0 and (arch, accelerator) in pruned_mappings.energy:
                logger.info(f"\t\tSkipping pruning of quantizated DNN {arch}: accuracy would be violated")
                continue

            # execute pruning and quantization
            compressor.reset()
            compressor.prune_and_quantize(prune_ratio, accelerator.precision)

            # evaluate the final pruned/quantized accuracy and sparsity statistics
            accuracy_stats = compressor.validate() if args.use_validation_set else compressor.test()
            total_stats, layer_stats = compressor.compute_model_statistics()
            logger.info(f"\t\tCompleted pruning/quantization for {arch}: ratio={prune_ratio}, bits={accelerator.precision}")
            logger.info("\t\tResults: " + ', '.join([
                f'{metric}={accuracy_stats[i]:.2f}'
                for i, metric in enumerate(compressor.accuracy_metrics)
            ]))
            logger.info(f"\t\tResults: sparsity={total_stats['sparsity']*100:.2f}%")

            # evaluate the DNN within timeloop
            energy = latency = 0
            optimizer.timeloop_wrapper.adjust_architecture(accelerator)
            for problem_name in optimizer.timeloop_problems_per_dnn[arch]:
                layer_name = optimizer.timeloop_problem_to_layer_name[arch][problem_name]

                # check extreme sparsity cases
                filters = optimizer.timeloop_wrapper.workloads[problem_name].config['instance']['M']
                channels = optimizer.timeloop_wrapper.workloads[problem_name].config['instance']['C']
                if (filters * (1 - layer_stats[layer_name]['sparsity_filters']) <= 1 and filters > 1) or \
                   (channels * (1 - layer_stats[layer_name]['sparsity_channels']) <= 1 and channels > 1):
                    logger.warning(f"\t\t{arch} layer {layer_name}: Attempting to leave less than 1 filter/channel "
                                   f"unpruned. Considering this layer completely pruned.")
                    continue

                # adjust for removed filters/channels
                optimizer.timeloop_wrapper.adjust_problem_dimension(
                    problem_name, 'M', adjust_by=(1 - layer_stats[layer_name]['sparsity_filters'])
                )
                optimizer.timeloop_wrapper.adjust_problem_dimension(
                    problem_name, 'C', adjust_by=(1 - layer_stats[layer_name]['sparsity_channels'])
                )
                # run timeloop and get results
                optimizer.timeloop_wrapper.run(problem_name)
                results = optimizer.timeloop_wrapper.get_results()
                energy += results.energy
                latency += results.cycles

            # save DNN results
            pruned_mappings.energy[(arch, accelerator)] = energy
            pruned_mappings.latency[(arch, accelerator)] = latency
            pruned_mappings.edp[(arch, accelerator)] = energy * latency

        if accelerator not in pruned_mappings.area:
            pruned_mappings.area[accelerator] = getattr(results, 'area', 0.0)

    logger.info(f"{'Skipped' if skip_mapping else 'Completed'} pruned mapping evaluation")
    savefile = os.path.join(optimizer.logdir, 'pruned_mappings.pkl')
    with open(savefile, 'wb') as f:
        pickle.dump(pruned_mappings, f)
    logger.info(f"Saved pruned mappings in {savefile}")

    total_area = sum([
        pruned_mappings.area[single_accel] for single_accel in hetero_accel
    ])

    # execute final schedule
    schedule = optimizer.scheduler.run(items=list(compressors.keys()),
                                       bins=hetero_accel,
                                       cost_dict=pruned_mappings.edp,
                                       weight_dict=pruned_mappings.latency,
                                       solver_type=args.solver_type)
    assert schedule is not None, f"Could not find valid final schedule!"

    # get results for energy and latency based on the final schedule
    total_energy = sum([
        pruned_mappings.energy[(entry.tag, entry.bin)] for entry in schedule.entries
    ])
    total_latency = max([
        sum([
            pruned_mappings.latency[(entry.tag, entry.bin)] for entry in entries
        ]) for bin, entries in schedule.as_dict(main_key='bin').items()
    ])

    # log the results of the scheduling
    schedule_str = '\n\t'.join([f'{entry.tag} -> {entry.bin}' for entry in schedule.entries])
    logger.info(f"Scheduler results:\n\t{schedule_str}")
    metrics = SimpleNamespace(energy=total_energy,
                                latency=total_latency,
                                edp=total_energy * total_latency,
                                area=total_area)
    logger.info(f"=> Final schedule metrics:")
    for metric, value in vars(metrics).items():
        logger.info(f"\t{metric.capitalize()} = {value:.3e}")

    return schedule, metrics


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

