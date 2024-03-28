import pandas as pd
import numpy as np
from enum import Enum
from matplotlib import rcParams, ticker
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
import argparse
import re
import os.path
import string
import os
from src import project_dir


def cfg_each_subplot(axes, group_names, heights,
                     bar_width=None, space_between_bars=0.4,
                     tick_label_size=12, in_group_labels=None, xticklabels=None,
                     colors=None, hatches=None):
    """
    Fully generic and parametrizable barblot
    Args:
        axes: axis objects to draw. Provide as many axes as different bars in a group
        group_names: the keys (names) of the different groups of bars
        heights: a list of lists. The outter list has as many items as bars in one group.
                 The inner list has as many items as there are number of groups
        bar_width: width of the bar in percentage
        space_between_bars: space between the groups
        labels: label each bar in the group
        tick_label_size, xticklabels, colors, hatches: self explanatory
    Returns:
        Bar containers, returned from plt.bar()
    """
    assert len(in_group_labels) == len(heights), f'Mismatch number of labels ({len(in_group_labels)}) ' \
                                                 f'and metrics ({len(heights)})'

    align = 'center' if len(heights) % 2 != 0 else 'edge'
    x = np.arange(len(group_names))  # x-positions of middle bar
    num_of_bars = len(heights)
    width = bar_width or (1 - space_between_bars) / num_of_bars

    # configure number of axes
    assert isinstance(axes, Axes) or \
        (isinstance(axes, (list, np.ndarray)) and all(isinstance(ax, Axes) for ax in axes)), \
        'Axes must be either an array of mpl.axes.Axes objects, or a single Axes object'
    if isinstance(axes, (list, np.ndarray)):
        assert len(axes) == len(in_group_labels), 'Number of axes must be equal to the number of bars in a group'
        axis_i = iter(axes)
    else:
        axis_i = None
        ax = axes

    all_bar_containers = []
    for bar_idx, (metric, label) in enumerate(zip(heights, in_group_labels)):
        if axis_i is not None:
            ax = next(axis_i)

        bar_cont = ax.bar(
            x=(x - (num_of_bars//2) * width) + bar_idx * width,
            height=[metric[idx] for idx, group in enumerate(group_names)],
            width=width,
            align=align,
            color=colors[bar_idx] if colors is not None else None,
            hatch=hatches[bar_idx] if hatches is not None else None,
            edgecolor='black',
            label=label,
            zorder=1
         )
        all_bar_containers.append(bar_cont)

        # ax.semilogy()
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, horizontalalignment='right')
        ax.tick_params(axis='x', which='both', length=0, labelsize=tick_label_size)#, labelrotation=45)
        ax.tick_params(axis='y', which='both', direction='in', labelsize=tick_label_size)
        ax.yaxis.grid(which="major", linestyle="-")
        # ax.yaxis.grid(which="minor", linestyle="--", alpha=.25)

    return all_bar_containers


class DisplayMode(Enum):
    ColumnWidth = 1
    TextWidth = 2


def display_mode_arg(argstr):
    str_to_display_mode_type = {str(entry.name).lower(): entry for entry in DisplayMode}
    try:
        return str_to_display_mode_type[argstr.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"--display-mode argument must be one of the following: "
                                         f"{str_to_display_mode_type.keys()}. Invalid argument {argstr}")


def main():
    # example call: 
    # python3 src/evaluation/comp_sota_per_metric.py \
    #      --resfile results/data/sota_simba.csv \
    #      --display-mode columnwidth \
    #      --pdf-name results/figures/comp_sota_eyeriss.pdf
    parser = argparse.ArgumentParser('Comparative evaluation with state of the art')
    parser.add_argument('--results-file', '--resfile', 
                        default=os.path.join(project_dir, 'results', 'data', 'sota_eyeriss.csv'),
                        help='File where results are stored')
    parser.add_argument('--single-metric', choices=['Area', 'Energy', 'Latency', 'EDP'],
                        help='Choose a single metric to produce a single sub-figure')
    parser.add_argument('--rows', dest='nrows', type=int, choices=[1, 4], default=1,
                        help='Number of rows for the subfigures (default is 1)')
    parser.add_argument('--display-mode', type=display_mode_arg, default='columnwidth',
                        help='Specify the display mode of the figure. Default is \'columnwidth\'')
    parser.add_argument('--pdf-name',
                        default=os.path.join(project_dir, 'results', 'figures', 'comp_sota_per_metric.pdf'),
                        help='File path to save the figure in PDF format')
    args = parser.parse_args()

    ##### Plot parameters #####

    # full textwidth is 7.13
    # columnwidth is 3.54
    if args.display_mode == DisplayMode.ColumnWidth:
        figx_inches = 3.54
        figy_inches = 1.6
    elif args.display_mode == DisplayMode.TextWidth:
        figx_inches = 7.13
        figy_inches = 3

    font_size = 7
    tick_label_size = 6

    bar_width = None
    space_between_bars = 0.3
    space_between_subfigures = 0.1

    colors_per_method = ['lightsteelblue', 'darkred', 'seagreen']
    # hatches = ['//', 'xx', '..']
    hatches = None
    normalize_wrt = 'Baseline'

    sota_citation = 2

    ############################

    assert re.search('[.]csv$', args.results_file)
    df = pd.read_csv(args.results_file)

    # check for compliance with known evaluation settings
    supported_accelerators = ['Eyeriss', 'Simba']
    evaluated_methods = ['Baseline', 'SOTA', 'Ours']
    method_legend_names = {'Ours': 'ARTEMIS (Ours)', 'Baseline': 'Baseline', 'SOTA': f'HDA-Q [{sota_citation}]'}
    # evaluated_metrics = ['Energy', 'Area', 'Latency', 'EDP']
    evaluated_metrics = ['Energy', 'Latency', 'EDP']
    workloads = ['ClassifiersA', 'ClassifiersB']
    if args.single_metric is not None:
        evaluated_metrics = [args.single_metric]
    csv_columns = ['WorkloadID', 'AcceleratorType', 'Method', 'Energy', 'Area', 'Latency', 'EDP']
    assert all(method in evaluated_methods for method in df['Method'].unique())
    assert all(column in csv_columns for column in df.columns)
    assert all(accelerator in supported_accelerators for accelerator in df['AcceleratorType'].unique())
    assert all(workload in workloads for workload in df['WorkloadID'].unique())

    total_subfigures = len(evaluated_metrics)
    nrows = min(args.nrows, total_subfigures) # in total, 4 subfigures (1 per metric)

    group_names = [str(workload_id).replace("Classifiers", "") for workload_id in workloads]
    in_group_labels = [method_legend_names[method] for method in evaluated_methods]

    # font ans latex parameters
    rcParams.update({
        "figure.figsize": (figx_inches, figy_inches),
        "font.size": str(font_size),
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}",
        "font.family": "lmodern",
        "font.serif": ["Computer Modern Roman"],
        "axes.axisbelow": True
    })
    letters = string.ascii_lowercase if args.single_metric is None else ['']

    fig, axes = plt.subplots(nrows=nrows,
                             ncols=total_subfigures // nrows)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    fig.tight_layout()
    plt.subplots_adjust(hspace=space_between_subfigures)

    for ax, metric, letter in zip(axes, evaluated_metrics, letters):
        # bar heights for each method, on the specified metric
        heights = {
            method: {
                workload_id: df.loc[(df['Method'] == method) & (df['WorkloadID'] == workload_id)][metric].iloc[0]
                for workload_id in workloads
            }
            for method in evaluated_methods
        }
        normalized_heights = [
            [
                heights[method][workload_id] / heights[normalize_wrt][workload_id]
                for workload_id in workloads
            ]
            for method in evaluated_methods
        ]
        cfg_each_subplot(axes=ax,
                         group_names=group_names,
                         heights=normalized_heights,
                         bar_width=bar_width,
                         space_between_bars=space_between_bars,
                         tick_label_size=tick_label_size,
                         in_group_labels=in_group_labels,
                         colors=colors_per_method,
                         hatches=hatches,)

        # ax.set_ylim(top=1.2)
        ax.semilogy()
        # ax.axhline(1, color='red', linestyle='--', zorder=0)

        # subfigure title
        metric = metric.capitalize() if metric.upper() != 'EDP' else 'EDP'
        title_text = rf'\textbf{{({letter}) Norm. {metric}}}' if letter != '' else \
                     rf'\textbf{{Norm. {metric}}}'
        ax.set_title(title_text, fontdict={'fontsize': str(font_size)})

    # legend
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        ncol=3,
        labelspacing=-3,
        columnspacing=1,
        handletextpad=0.5,
        edgecolor=None,
        framealpha=0,
        fontsize=font_size,
        bbox_to_anchor=(0.52, -0.08), loc='lower center'
    )

    # save plot
    os.makedirs(os.path.join(project_dir, 'results', 'figures'), exist_ok=True)
    fig.savefig(args.pdf_name, bbox_inches='tight', pad_inches=0)
    print(f"Barplot was saved in {args.pdf_name}")


if __name__ == "__main__":
    main()
