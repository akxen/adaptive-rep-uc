"""Plotting model results"""

import os
import pickle

import MySQLdb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogLocator, NullFormatter

from processing import Results


def cm_to_in(centimeters):
    """Convert inches to cm"""
    return centimeters * 0.393701


def calibration_intervals_plot(output_dir):
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

    fig.savefig(os.path.join(output_dir, 'calibration_intervals.pdf'), transparent=True)
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


def revenue_target_plot(output_dir):
    """Plot baseline and cumulative scheme revenue when a non-zero revenue target is specified"""

    # Get baselines and cumulative scheme revenue
    baselines, revenue = process.get_baselines_and_revenue('revenue_target')
    revenue = revenue.div(1e6)

    fig, ax = plt.subplots()
    axc = ax.twinx()
    ax.plot(range(1, 53), baselines.values, color='red', alpha=0.5, linewidth=1.2)
    axc.plot(range(1, 53), revenue.values, color='blue', alpha=0.5, linewidth=1.2)
    # axc.ticklabel_format(style='sci', scilimits=(-2, 2), useMathText=True)
    ax.tick_params(labelsize=7)
    axc.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(MultipleLocator(12))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.002))
    axc.yaxis.set_minor_locator(MultipleLocator(0.5))
    axc.yaxis.set_major_locator(MultipleLocator(2.5))
    axc.yaxis.offsetText.set_fontsize(8)
    ax.set_xlabel('Week', fontsize=9)
    ax.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=9, labelpad=-0.07)
    axc.set_ylabel('Revenue (M\$)', fontsize=9, labelpad=-0.07)
    axc.plot([0, 52], [0, 0], linestyle='--', color='k', alpha=0.5, linewidth=0.8)
    axc.plot([0, 52], [10, 10], linestyle='--', color='k', alpha=0.5, linewidth=0.8)

    lns = ax.lines + axc.lines
    ax.legend(lns, ['baseline', 'revenue'], fontsize=9, ncol=2, loc='upper left', bbox_to_anchor=(0, 1.13, 0, 0),
              frameon=False, )

    fig.set_size_inches(cm_to_in(12), cm_to_in(7))
    fig.subplots_adjust(left=0.11, bottom=0.15, right=0.89, top=0.92)

    fig.savefig(os.path.join(output_dir, 'revenue_targeting.png'))
    fig.savefig(os.path.join(output_dir, 'revenue_targeting.pdf'), transparent=True)
    plt.show()


def emissions_intensity_shock_plot(output_dir):
    """Plot baseline and revenue from emissions intensity shock"""

    # Get baselines and cumulative scheme revenue
    baselines, revenue = process.get_baselines_and_revenue('emissions_intensity_shock')
    revenue = revenue.div(1e6)

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(range(1, 53), revenue.values, color='blue', alpha=0.5, linewidth=1.2)
    ax2.plot(range(1, 53), baselines.values, color='red', alpha=0.5, linewidth=1.2)

    ax1.plot([10, 10], [-100, 10], linewidth=1.2, linestyle='--', alpha=0.5, color='black')
    ax1.set_ylim([-95, 5])

    ax2.plot([10, 10], [0.1, 1.1], linewidth=1.2, linestyle='--', alpha=0.5, color='black')
    ax2.set_ylim([0.12, 1.08])

    ax1.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    ax2.xaxis.set_major_locator(MultipleLocator(12))
    ax2.xaxis.set_minor_locator(MultipleLocator(2))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.yaxis.set_major_locator(MultipleLocator(25))
    ax1.yaxis.set_minor_locator(MultipleLocator(12.5))

    ax2.set_xlabel('Week', fontsize=9)
    ax1.set_ylabel('Revenue (M\$)', fontsize=8, labelpad=-0.07)
    ax2.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=8, labelpad=-0.07)

    lns = [ax2.lines[0], ax1.lines[0]]
    ax2.legend(lns, ['baseline', 'revenue'], fontsize=8, ncol=2, loc='upper right', bbox_to_anchor=(1, 1, 0, 0),
               frameon=False)

    fig.set_size_inches(cm_to_in(12), cm_to_in(7))
    fig.subplots_adjust(left=0.09, bottom=0.15, right=0.99, top=0.99)

    fig.savefig(os.path.join(output_dir, 'emissions_intensity_shock.png'))
    fig.savefig(os.path.join(output_dir, 'emissions_intensity_shock.pdf'), transparent=True)
    plt.show()


def price_comparison_plot(output_dir):
    """Compare BAU, carbon tax, and REP average weekly prices"""

    p_bau, _ = process.get_week_prices('bau')
    p_tax, _ = process.get_week_prices('carbon_tax')
    p_rep, _ = process.get_week_prices('4_calibration_intervals')

    fig, ax = plt.subplots()
    ax.plot(range(1, 53), p_bau.values, color='red', alpha=0.7)
    ax.plot(range(1, 53), p_tax.values, color='#159eab', alpha=0.7)
    ax.plot(range(1, 53), p_rep.values, color='#15ab49', alpha=0.7)
    ax.legend(['BAU', 'Tax', 'REP'], fontsize=8, ncol=3, loc='upper right', bbox_to_anchor=(1, 1.01, 0, 0),
              frameon=False)

    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(MultipleLocator(12))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2))

    ax.set_xlabel('Week', fontsize=9)
    ax.set_ylabel('Average price ($/MWh)', fontsize=9)

    fig.set_size_inches(cm_to_in(12), cm_to_in(7))
    fig.subplots_adjust(left=0.09, bottom=0.15, right=0.99, top=0.99)

    fig.savefig(os.path.join(output_dir, 'price_comparison.png'))
    fig.savefig(os.path.join(output_dir, 'price_comparison.pdf'), transparent=True)

    plt.show()


def quantile_regression_plot(output_dir):
    """Plot quantile regression results"""

    # Load quantile regression results
    with open(os.path.join(output_dir, 'quantile_regression_results.pickle'), 'rb') as f:
        results = pickle.load(f)

    # Observed values up until point forecast is made (last interval)
    y_obs = results['dataset'].loc[:, 'lag_0'].values[-10:]
    x_obs = range(1, len(y_obs) + 1)

    # Get quantile regression results
    q_reg = results['results']

    # Plot lines for each quantile and future period
    fig, ax = plt.subplots()
    for k in q_reg.keys():
        x = range(max(x_obs), max(x_obs) + 5)
        y = [y_obs[-1]] + list(q_reg[k].values())
        ax.plot(x, y, color='grey', linewidth=0.6, linestyle='--', alpha=0.5)

    # Fill area between curves
    pairs = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
    for pair in pairs:
        x = range(max(x_obs), max(x_obs) + 5)
        y1, y2 = [y_obs[-1]] + list(q_reg[pair[0]].values()), [y_obs[-1]] + list(q_reg[pair[1]].values())
        ax.fill_between(x, y1, y2, facecolor='#d43126', alpha=0.5)

    # Plot observed values up until point forecast is made (last data point)
    ax.plot(x_obs, y_obs, color='#264fd4', alpha=0.8)

    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(200))
    ax.set_xlabel('Week', fontsize=9)
    ax.set_ylabel('Energy (MWh)', fontsize=9)
    ax.text(1, 96000, 'LYA1')

    fig.set_size_inches(cm_to_in(12), cm_to_in(7))
    fig.subplots_adjust(left=0.13, bottom=0.15, right=0.99, top=0.99)

    fig.savefig(os.path.join(output_dir, 'quantile_regression_example.png'))
    fig.savefig(os.path.join(output_dir, 'quantile_regression_example.pdf'), transparent=True)

    plt.show()


if __name__ == '__main__':
    # Output directory
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Object used to process model results
    process = Results()

    # Compare baseline and revenue traces for different calibration intervals
    # calibration_intervals_plot()

    # Plot lagged dispatch
    # plot_lagged_dispatch(output_directory)

    # Plot revenue target
    # revenue_target_plot()

    # Plot revenue and baselines following an emissions intensity shock
    # emissions_intensity_shock_plot()

    # Plot price comparison
    # price_comparison_plot(output_directory)

    # Quantile regression plot
    quantile_regression_plot(output_directory)
