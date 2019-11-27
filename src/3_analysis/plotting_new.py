"""Plot model results"""

import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from processing import Results


class PlotUtils:
    def __init__(self):
        pass

    def load_data(self, use_cache=True):
        """Load model data"""
        pass

    def save_data(self):
        """Save data that is used when plotting"""
        pass


class CreatePlots:
    def __init__(self, output_dir):
        self.utils = PlotUtils()
        self.results = Results()

        # Directory containing output files / plots
        self.output_dir = output_dir

    @staticmethod
    def cm_to_in(cm):
        """Convert centimeters to inches"""

        return cm * 0.393701

    @staticmethod
    def load_cache(directory, filename):
        """Load cached results"""

        with open(os.path.join(directory, filename), 'rb') as f:
            results = pickle.load(f)

        return results

    @staticmethod
    def save(results, directory, filename):
        """Save results"""

        with open(os.path.join(directory, filename), 'wb') as f:
            pickle.dump(results, f)

    def get_calibration_interval_comparison_plot_data(self, use_cache=False, save=True):
        """Plot impact of difference calibration intervals"""

        # Filename
        filename = 'calibration_intervals_comparison_plot_data.pickle'

        # Load cached file if specified and return (faster loading)
        if use_cache:
            results = self.load_cache(self.output_dir, filename)

            return results

        # Results container
        results = {}

        for i in range(1, 7):
            # Construct case name
            case_name = f'{i}_calibration_intervals'

            # Get baselines and cumulative scheme revenue
            baselines, revenue = self.results.get_baselines_and_revenue(case_name)

            # Get generators subjected to scheme
            # TODO: Need to update this to extract generators based on MPC model parameters
            generators = self.results.data.get_thermal_unit_duids()

            # Weekly emissions intensity
            emissions_intensity = self.results.get_week_emissions_intensity(f'{i}_calibration_intervals', generators)

            # Append to container
            results[i] = {'baselines': baselines, 'revenue': revenue, 'emissions_intensity': emissions_intensity}

        # Save results
        if save:
            self.save(results, self.output_dir, filename)

        return results

    def plot_calibration_interval_comparison(self):
        """Plot baseline and revenue comparison when using different numbers of calibration intervals"""

        # Get data
        try:
            r = self.get_calibration_interval_comparison_plot_data(use_cache=True)
        except Exception as e:
            print(e)
            r = self.get_calibration_interval_comparison_plot_data(use_cache=False, save=True)

        # Initialise figure
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True)

        # Plot baselines
        baseline_style = {'color': '#225de6', 'alpha': 0.7, 'linewidth': 1}
        for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
            ax.plot(range(1, len(r[i + 1]['baselines'].values) + 1), r[i + 1]['baselines'].values, **baseline_style)

        # Plot scheme emissions intensity
        emissions_intensity_style = {'color': '#39850b', 'alpha': 0.7, 'linestyle': '--', 'linewidth': 1.1}
        for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
            ax.plot(range(1, len(r[i + 1]['emissions_intensity'].values) + 1), r[i + 1]['emissions_intensity'].values,
                    **emissions_intensity_style)

        # Create twin axes for revenue
        ax12 = ax1.twinx()
        ax22 = ax2.twinx()
        ax32 = ax3.twinx()
        ax42 = ax4.twinx()
        ax52 = ax5.twinx()
        ax62 = ax6.twinx()

        # Plot scheme revenue
        revenue_style = {'color': '#e63622', 'alpha': 0.8, 'linewidth': 1}
        for i, ax in enumerate([ax12, ax22, ax32, ax42, ax52, ax62]):
            ax.plot(range(1, len(r[i + 1]['revenue'].values) + 1), r[i + 1]['revenue'].values, **revenue_style)

        # Remove y-tick labels
        for ax in [ax12, ax22, ax42, ax52]:
            ax.axes.yaxis.set_ticklabels([])

        # Set common y-limit for y-axis
        for ax in [ax12, ax22, ax32, ax42, ax52, ax62]:
            ax.set_ylim([-3.5e6, 3.5e6])

        # Use scientific notation for revenue
        ax32.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax62.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

        # Format ticks
        ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))

        ax1.yaxis.set_major_locator(plt.MultipleLocator(0.02))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.01))

        for ax in [ax12, ax22, ax32, ax42, ax52, ax62]:
            ax.yaxis.set_major_locator(plt.MultipleLocator(2e6))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(1e6))

        # Change fontsize
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax32, ax62]:
            ax.tick_params(labelsize=7)

        ax32.yaxis.get_offset_text().set_size(7)
        ax62.yaxis.get_offset_text().set_size(7)

        ax32.yaxis.get_offset_text().set(ha='left')
        ax62.yaxis.get_offset_text().set(ha='left')

        # Define labels
        ax1.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=7, labelpad=-0.1)
        ax4.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=7, labelpad=-0.1)
        ax32.set_ylabel('Revenue ($)', fontsize=7, labelpad=-0.1)
        ax62.set_ylabel('Revenue ($)', fontsize=7, labelpad=-0.1)
        ax4.set_xlabel('Week', fontsize=7)
        ax5.set_xlabel('Week', fontsize=7)
        ax6.set_xlabel('Week', fontsize=7)

        # Add text to denote different calibration intervals
        ax1.set_title('1 calibration interval', fontsize=7, pad=-0.001)
        ax2.set_title('2 calibration intervals', fontsize=7, pad=-0.001)
        ax3.set_title('3 calibration intervals', fontsize=7, pad=-0.001)
        ax4.set_title('4 calibration intervals', fontsize=7, pad=-0.001)
        ax5.set_title('5 calibration intervals', fontsize=7, pad=-0.001)
        ax6.set_title('6 calibration intervals', fontsize=7, pad=-0.001)

        # Create legend
        baseline = ax2.lines[0]
        emissions = ax2.lines[1]
        lns = [baseline, emissions]
        labs = ['baseline', 'avg. emissions intensity']
        ax2.legend(lns, labs, fontsize=6, ncol=1, frameon=False, loc='lower left', bbox_to_anchor=(0.01, -0.025))
        ax22.legend(['revenue'], fontsize=6, ncol=1, frameon=False, loc='upper left', bbox_to_anchor=(0.01, 0.99))

        # Add text to denote sub-figures
        ax1.text(47, 0.967, 'a', fontsize=9, weight='bold')
        ax2.text(47, 0.967, 'b', fontsize=9, weight='bold')
        ax3.text(47, 0.967, 'c', fontsize=9, weight='bold')
        ax4.text(47, 0.967, 'd', fontsize=9, weight='bold')
        ax5.text(47, 0.967, 'e', fontsize=9, weight='bold')
        ax6.text(47, 0.967, 'f', fontsize=9, weight='bold')

        # Set figure size
        fig.set_size_inches((self.cm_to_in(16.5), self.cm_to_in(9)))

        # Adjust subplot position
        fig.subplots_adjust(left=0.08, top=0.95, right=0.93, wspace=0.15)

        # Save figures
        fig.savefig(os.path.join(self.output_dir, 'calibration_intervals_comparison.png'))
        fig.savefig(os.path.join(self.output_dir, 'calibration_intervals_comparison.pdf'))

        plt.show()

    def get_revenue_targeting_plot_data(self, use_cache=False, save=True):
        """Get data used to show impact of positive revenue target"""

        # Filename
        filename = 'revenue_targeting_plot_data.pickle'

        # Load cached file if specified and return (faster loading)
        if use_cache:
            results = self.load_cache(self.output_dir, filename)
            return results

        # Container for results
        results = {}

        for c in ['1_calibration_intervals', '4_calibration_intervals', 'revenue_target']:
            # Get baselines and cumulative scheme revenue
            baselines, revenue = self.results.get_baselines_and_revenue(c)

            # Get generators subjected to scheme
            # TODO: Need to update this to extract generators based on MPC model parameters
            generators = self.results.data.get_thermal_unit_duids()

            # Weekly emissions intensity
            emissions_intensity = self.results.get_week_emissions_intensity(c, generators)

            # Case parameters
            parameters = self.results.get_case_parameters(c)

            # Append to results container
            results[c] = {'baselines': baselines, 'revenue': revenue, 'emissions_intensity': emissions_intensity,
                          'parameters': parameters}

        # Save results
        if save:
            self.save(results, self.output_dir, filename)

        return results

    def plot_revenue_targeting(self):
        """Plot baseline and revenue comparison when using different numbers of calibration intervals"""

        # Get data
        try:
            r = self.get_revenue_targeting_plot_data(use_cache=True)
        except Exception as e:
            print(e)
            r = self.get_revenue_targeting_plot_data(use_cache=False, save=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)

        # Plot revenue neutral baselines with different calibration intervals
        short_ci_style = {'color': '#225de6', 'alpha': 0.7, 'linewidth': 1}
        ax1.plot(range(1, len(r['1_calibration_intervals']['baselines'].values) + 1),
                 r['1_calibration_intervals']['baselines'].values, **short_ci_style)

        long_ci_style = {'color': 'r', 'alpha': 0.7, 'linewidth': 1}
        ax1.plot(range(1, len(r['4_calibration_intervals']['baselines'].values) + 1),
                 r['4_calibration_intervals']['baselines'].values, **long_ci_style)

        emissions_intensity_style = {'color': 'g', 'alpha': 0.7, 'linewidth': 0.7, 'linestyle': '--'}
        ax1.plot(range(1, len(r['4_calibration_intervals']['emissions_intensity'].values) + 1),
                 r['4_calibration_intervals']['emissions_intensity'].values, **emissions_intensity_style)

        # Plot price targeting baselines
        ax2.plot(range(1, len(r['revenue_target']['baselines'].values) + 1),
                 r['revenue_target']['baselines'].values, **long_ci_style)

        ax2.plot(range(1, len(r['4_calibration_intervals']['emissions_intensity'].values) + 1),
                 r['4_calibration_intervals']['emissions_intensity'].values, **emissions_intensity_style)

        # Plot revenue neutral target paths with different calibration intervals
        ax3.plot(range(1, len(r['1_calibration_intervals']['revenue'].values) + 1),
                 r['1_calibration_intervals']['revenue'].values, **short_ci_style)

        ax3.plot(range(1, len(r['4_calibration_intervals']['revenue'].values) + 1),
                 r['4_calibration_intervals']['revenue'].values, **long_ci_style)

        # Plot baseline when positive revenue target is implemented
        ax4.plot(range(1, len(r['revenue_target']['revenue'].values) + 1),
                 r['revenue_target']['revenue'].values, **long_ci_style)

        # Plot revenue targets
        target_style = {'color': 'k', 'alpha': 0.7, 'linewidth': 0.7, 'linestyle': '--'}

        ax3.plot(range(1, 53), [r['1_calibration_intervals']['parameters']['revenue_target'][2018][i] for i in range(1, 53)],
                 **target_style)

        ax4.plot(range(1, 53), [r['revenue_target']['parameters']['revenue_target'][2018][i] for i in range(1, 53)],
                 **target_style)

        # Set axis limit
        ax1.set_ylim([0.96, 1.03])
        ax2.set_ylim([0.96, 1.03])

        ax3.set_ylim([-0.5e7, 1.3e7])
        ax4.set_ylim([-0.5e7, 1.3e7])

        # Shade region
        ax2.fill_between([10, 20], [0, 0], [2, 2], color='k', alpha=0.2, linewidth=0)
        ax4.fill_between([10, 20], [-1e7, -1e7], [2e7, 2e7], color='k', alpha=0.2, linewidth=0)

        # Change tick label size
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(labelsize=7)

        # Remove tick labels
        ax2.axes.yaxis.set_ticklabels([])
        ax4.axes.yaxis.set_ticklabels([])

        # Set major and minor ticks
        for ax in [ax1, ax2]:
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))

        ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))

        for ax in [ax3, ax4]:
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.5e7))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25e7))

        # Add legend
        lns = ax3.lines + [ax1.lines[-1], ax4.lines[-1]]
        labs = ['1 interval', '2  intervals', 'revenue target', 'avg. emissions intensity']
        ax3.legend(lns, labs, frameon=False, fontsize=5, loc=2, bbox_to_anchor=(-0.008, 1))

        # Plot titles
        ax1.set_title('Revenue neutral', fontsize=7)
        ax2.set_title('Positive revenue target', fontsize=7)
        ax1.title.set_position((ax1.title.get_position()[0], ax1.title.get_position()[1] - 0.05))
        ax2.title.set_position((ax2.title.get_position()[0], ax2.title.get_position()[1] - 0.05))

        # Add axes labels
        ax3.set_xlabel('Week', fontsize=7)
        ax4.set_xlabel('Week', fontsize=7)

        ax1.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=7)
        ax3.set_ylabel('Revenue ($)', fontsize=7)

        # Add text to denote sub-figures
        ax1.text(47, 0.967, 'a', fontsize=9, weight='bold')
        ax2.text(47, 0.967, 'b', fontsize=9, weight='bold')
        ax3.text(47, -0.35e7, 'c', fontsize=9, weight='bold')
        ax4.text(47, -0.35e7, 'd', fontsize=9, weight='bold')

        # Change size of scientific notation
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax3.yaxis.get_offset_text().set_size(7)
        ax3.yaxis.get_offset_text().set(ha='right', va='center')

        # Set figure size
        fig.set_size_inches((self.cm_to_in(7.8), self.cm_to_in(7.8)))

        # Adjust subplot position
        fig.subplots_adjust(left=0.17, top=0.95, bottom=0.12, right=0.99, wspace=0.12, hspace=0.18)

        # Save figures
        fig.savefig(os.path.join(self.output_dir, 'revenue_targeting.png'))
        fig.savefig(os.path.join(self.output_dir, 'revenue_targeting.pdf'))

        plt.show()

    def get_revenue_floor_plot_data(self, use_cache=False, save=True):
        """Get data for revenue lower bound plot"""

        filename = 'revenue_floor_data.pickle'

        # Load cached file if specified and return (faster loading)
        if use_cache:
            results = self.load_cache(self.output_dir, filename)
            return results

        # Results container
        results = {}

        for c in ['revenue_floor', '3_calibration_intervals']:
            # Get baselines and cumulative scheme revenue
            baselines, revenue = self.results.get_baselines_and_revenue(c)

            # Get generators subjected to scheme
            # TODO: Need to update this to extract generators based on MPC model parameters
            generators = self.results.data.get_thermal_unit_duids()

            # Weekly emissions intensity
            emissions_intensity = self.results.get_week_emissions_intensity(c, generators)

            # Combine results
            results[c] = {'baselines': baselines, 'revenue': revenue, 'emissions_intensity': emissions_intensity}

        # Save results
        if save:
            self.save(results, self.output_dir, filename)

        return results


if __name__ == '__main__':
    output_directory = os.path.join(os.path.dirname(__file__), 'output', 'figures')

    # Object used to create plots
    plots = CreatePlots(output_directory)

    # Plot baseline and revenue for different calibration interval durations
    # plots.plot_calibration_interval_comparison()

    # Price targeting plot
    # r = plots.get_revenue_targeting_plot_data(use_cache=True)
    # plots.plot_revenue_targeting()

    # Revenue floor plot
    r = plots.get_revenue_floor_plot_data()
