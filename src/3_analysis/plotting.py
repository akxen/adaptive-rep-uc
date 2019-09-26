"""Plotting model results"""

import os
import pickle

import MySQLdb
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogLocator, NullFormatter

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


def plot_lagged_dispatch(output_dir):
    """Compare dispatch for different lag lengths"""

    with open(os.path.join(output_dir, 'dispatch_weekly.pickle'), 'rb') as f:
        dispatch = pickle.load(f)

    def plot_week_energy(df, future_week, ax):
        """Plot energy in current week compared to future weeks"""
        # Set all values less than 1 equal to zero so they can be resolved on a log-log plot
        df[df < 1] = 1

        # Convert to dictionary
        energy = df.to_dict()

        # Future energy values
        future = df.shift(-future_week).to_dict()

        # Compare future week with current week
        comparison = {g: {week: (value, future[g][week]) for week, value in d.items()} for g, d in energy.items()}

        # Extract tuples comparing values between current week and specified future week
        pairs = [j for i in [list(i.values()) for i in comparison.values()] for j in i]

        # Construct scatter plot
        ax.scatter([i[0] for i in pairs], [i[1] for i in pairs], alpha=0.3, s=1, color='#c93838')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([0.8, 300000])
        ax.set_ylim([0.8, 300000])
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Energy', fontsize=8, labelpad=-0.1)

        locmajx = LogLocator(base=10, numticks=10)
        ax.xaxis.set_major_locator(locmajx)

        locmajy = LogLocator(base=10, numticks=8)
        ax.yaxis.set_major_locator(locmajy)

        locminx = LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
        ax.xaxis.set_minor_locator(locminx)
        ax.xaxis.set_minor_formatter(NullFormatter())

        locminy = LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
        ax.yaxis.set_minor_locator(locminy)
        ax.yaxis.set_minor_formatter(NullFormatter())

        if future_week == 1:
            ax.set_ylabel(f'Energy +{future_week} week', fontsize=8, labelpad=0)
        else:
            ax.set_ylabel(f'Energy +{future_week} weeks', fontsize=8, labelpad=0)

        ax.plot([0, 200000], [0, 200000], color='k', linestyle='--', linewidth=1)

        return ax

    # Construct figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    plot_week_energy(dispatch, 1, ax1)
    plot_week_energy(dispatch, 2, ax2)
    plot_week_energy(dispatch, 4, ax3)
    plot_week_energy(dispatch, 6, ax4)

    fig.set_size_inches(cm_to_in(12), cm_to_in(7))
    fig.subplots_adjust(hspace=0.4, wspace=0.25, left=0.08, bottom=0.12, top=0.99, right=0.99)

    fig.savefig(os.path.join(output_dir, 'lagged_dispatch.pdf'), transparent=True)
    fig.savefig(os.path.join(output_dir, 'lagged_dispatch.png'), transparent=True)

    plt.show()


if __name__ == '__main__':
    # Output directory
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Object used to process model results
    process = Results()

    # Compare baseline and revenue traces for different calibration intervals
    # calibration_intervals_plot()

    plot_lagged_dispatch(output_directory)


