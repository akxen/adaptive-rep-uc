"""Plotting model results"""

import os

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

from processing import Results


def cm_to_in(centimeters):
    """Convert inches to cm"""
    return centimeters * 0.393701


def calibration_intervals_plot():
    """Analyse baseline and scheme revenue for different calibration intervals"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    ax1c = ax1.twinx()
    ax2c = ax2.twinx()
    ax3c = ax3.twinx()
    ax4c = ax4.twinx()

    ax1c.get_shared_y_axes().join(ax1c, ax2c)
    ax3c.get_shared_y_axes().join(ax3c, ax4c)
    ax1.get_shared_y_axes().join(ax1, ax3)
    ax2c.get_shared_y_axes().join(ax2c, ax4c)

    def plot_baseline_revenue_subplot(case_name, ax, axc):
        """Plot data for a given case. Return figure and axes objects"""

        # Get baselines and cumulative scheme revenue
        baselines, revenue = process.get_baselines_and_revenue(case_name)

        # Plot baselines and scheme revenue
        ax.plot(range(1, 53), baselines.values, color='red', alpha=0.5, linewidth=0.8)
        axc.plot(range(1, 53), revenue.values, color='blue', alpha=0.5, linewidth=0.8)
        axc.ticklabel_format(style='sci', scilimits=(-2, 2), useMathText=True)
        ax.tick_params(labelsize=7)
        axc.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(3))
        ax.yaxis.set_minor_locator(MultipleLocator(0.005))
        axc.yaxis.set_minor_locator(MultipleLocator(500000))
        axc.yaxis.offsetText.set_fontsize(8)

        return ax, axc

    ax1, ax1c = plot_baseline_revenue_subplot('1_calibration_intervals', ax1, ax1c)
    ax2, ax2c = plot_baseline_revenue_subplot('2_calibration_intervals', ax2, ax2c)
    ax3, ax3c = plot_baseline_revenue_subplot('4_calibration_intervals', ax3, ax3c)
    ax4, ax4c = plot_baseline_revenue_subplot('6_calibration_intervals', ax4, ax4c)

    fig.set_size_inches(cm_to_in(11.5), cm_to_in(7.5))
    fig.subplots_adjust(left=0.11, wspace=0.12, top=0.94, hspace=0.22, bottom=0.15)
    ax1c.set_yticklabels([])
    ax3c.set_yticklabels([])
    ax1.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=7, labelpad=-0.005)
    ax3.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=7, labelpad=-0.005)
    ax2c.set_ylabel('Revenue ($)', fontsize=7)
    ax4c.set_ylabel('Revenue ($)', fontsize=7)
    ax3.set_xlabel('Week', fontsize=7)
    ax4.set_xlabel('Week', fontsize=7)
    ax1.text(37, 0.965, '1 interval', fontsize=6)
    ax2.text(37, 0.965, '2 intervals', fontsize=6)
    ax3.text(37, 0.965, '4 intervals', fontsize=6)
    ax4.text(37, 0.965, '6 intervals', fontsize=6)

    lns = ax1.lines + ax1c.lines
    ax1.legend(lns, ['baseline', 'revenue'], fontsize=6, ncol=2, loc='upper left', bbox_to_anchor=(0, 1.18, 0, 0),
               frameon=False)

    fig.savefig('output/calibration_intervals.pdf', transparent=True)
    plt.show()


if __name__ == '__main__':
    # Object used to process model results
    process = Results()

    # Compare baseline and revenue traces for different calibration intervals
    calibration_intervals_plot()


