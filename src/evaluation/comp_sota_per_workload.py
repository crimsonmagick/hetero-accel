import pandas as pd
import numpy as np
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
        heights: a list of lists. The outter list has as many items as different groups.
                 Each item is a group of bars. The item contains the heights
                 of each bar in the group. Meaning each inner item (list) has as many items
                 as different bars in a group
        bar_width: width of the bar in percentage
        space_between_bars: space between the groups
        labels: label each bar in the group
        tick_label_size, xticklabels, colors, hatches: self explanatory
    Returns:
        Bar containers, returned from plt.bar()
    """
    assert len(in_group_labels) == len(heights), 'Mismatch number of labels and metrics'

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

        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, horizontalalignment='right')
        ax.tick_params(axis='x', which='both', length=0, labelsize=tick_label_size, labelrotation=45)
        ax.tick_params(axis='y', which='both', direction='in', labelsize=tick_label_size)
        # ax.yaxis.grid(which="major", linestyle="-")
        # ax.yaxis.grid(which="minor", linestyle="--", alpha=.25)

    return all_bar_containers


def main():
    parser = argparse.ArgumentParser('Comparative evaluation with state of the art')
    parser.add_argument('--results-file', '--resfile', default=os.path.join(project_dir, 'src', 'evaluation', 'sota.csv'),
                        help='File where results are stored')
    parser.add_argument('--single-workload', type=int, choices=[1, 2, 3],
                        help='Choose a single workload to produce a single sub-figure')
    parser.add_argument('--rows', dest='nrows', type=int, default=1,
                        help='Number of rows for the subfigures (default is 1)')
    args = parser.parse_args()

    ##### Plot parameters #####

    # full textwidth is 7.13
    figx_inches = 7.13
    figy_inches = 3

    font_size = 12
    tick_label_size = 12

    bar_width = 0.1
    space_between_bars = 0.2
    space_between_subfigures = 0.5

    colors_per_metric = ['darkgreen', 'lightsteelblue', 'bisque', 'grey']
    hatches = ['//', 'xx', '..', '--']
    normalize_wrt = 'Baseline'

    ############################

    assert re.search('[.]csv$', args.results_file)
    df = pd.read_csv(args.results_file)

    # check for compliance with known evaluation settings
    supported_accelerators = ['Eyeriss', 'Simba']
    evaluated_methods = ['Ours', 'Baseline', 'SOTA']
    evaluated_metrics = ['Energy', 'Area', 'Latency', 'EDP']
    workloads = [1, 2, 3]
    if args.single_workload is not None:
        workloads = [args.single_workload]
    csv_columns = ['WorkloadID', 'AcceleratorType', 'Method', 'Energy', 'Area', 'Latency', 'EDP']
    assert all(method in evaluated_methods for method in df['Method'].unique())
    assert all(column in csv_columns for column in df.columns)
    assert all(accelerator in supported_accelerators for accelerator in df['AcceleratorType'].unique())

    total_subfigures = len(workloads)
    nrows = min(args.nrows, total_subfigures) # in total, 4 subfigures (1 per metric)
    group_names = [method for method in evaluated_methods if method != normalize_wrt]

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
    letters = string.ascii_lowercase if args.single_workload is None else ['']

    fig, axes = plt.subplots(nrows=nrows,
                             ncols=total_subfigures // nrows)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    fig.tight_layout()
    plt.subplots_adjust(hspace=space_between_subfigures)

    for ax, workload, letter in zip(axes, workloads, letters):
        # bar heights for each metric, on the specified workload
        unormalized_heights = {
            method: {
                metric: df.loc[(df['Method'] == method) & (df['WorkloadID'] == workload)][metric].iloc[0]
                for metric in evaluated_metrics
            }
            for method in evaluated_methods
        }
        normalized_heights = [
            [
                unormalized_heights[method][metric] / unormalized_heights[normalize_wrt][metric]
                for method in group_names
            ]
            for metric in evaluated_metrics
        ]

        cfg_each_subplot(axes=ax,
                         group_names=group_names,
                         heights=normalized_heights,
                         bar_width=bar_width,
                         space_between_bars=space_between_bars,
                         tick_label_size=tick_label_size,
                         in_group_labels=evaluated_metrics,
                         colors=colors_per_metric,
                         hatches=hatches,)

        # subfigure title
        title_text = rf'\textbf{{({letter}) W{workload}}}' if letter != '' else \
                     rf'\textbf{{W{workload}}}'
        ax.set_title(title_text, fontdict={'fontsize': str(font_size)})

    # legend
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        ncol=len(evaluated_metrics),
        labelspacing=-3,
        columnspacing=1,
        handletextpad=0.5,
        edgecolor=None,
        framealpha=0,
        fontsize=font_size,
        bbox_to_anchor=(0.52, -0.05), loc='lower center'
    )

    # save plot
    os.makedirs(os.path.join(project_dir, 'results', 'figures'), exist_ok=True)
    savefile = os.path.join(project_dir, 'results', 'figures', 'comp_sota_per_workload.pdf')
    fig.savefig(savefile, bbox_inches='tight', pad_inches=0)
    print(f"Barplot was saved in {savefile}")


if __name__ == "__main__":
    main()
