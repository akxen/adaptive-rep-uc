"""Plot model results"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'base'))

import pickle

import MySQLdb
import MySQLdb.cursors
import numpy as np
import pandas as pd
import matplotlib.ticker
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

from processing import Results
from forecast import MonteCarloForecast


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

        # Mapping for standardised colors
        self.colours = {'red': '#f21d24', 'blue': '#4262ed', 'green': '#44ba3a', 'purple': '#a531cc',
                        'orange': '#e8aa46'}

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

        ax32.yaxis.get_offset_text().set_size(8)
        ax62.yaxis.get_offset_text().set_size(8)

        ax32.yaxis.get_offset_text().set(ha='left')
        ax62.yaxis.get_offset_text().set(ha='left')

        # Define labels
        ax1.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=8, labelpad=0)
        ax4.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=8, labelpad=0)
        ax32.set_ylabel('Revenue ($)', fontsize=8, labelpad=1)
        ax62.set_ylabel('Revenue ($)', fontsize=8, labelpad=1)
        ax4.set_xlabel('Week', fontsize=8)
        ax5.set_xlabel('Week', fontsize=8)
        ax6.set_xlabel('Week', fontsize=8)

        # Add text to denote different calibration intervals
        ax1.set_title('1 calibration interval', fontsize=8, pad=-0.001)
        ax2.set_title('2 calibration intervals', fontsize=8, pad=-0.001)
        ax3.set_title('3 calibration intervals', fontsize=8, pad=-0.001)
        ax4.set_title('4 calibration intervals', fontsize=8, pad=-0.001)
        ax5.set_title('5 calibration intervals', fontsize=8, pad=-0.001)
        ax6.set_title('6 calibration intervals', fontsize=8, pad=-0.001)

        # Create legend
        baseline = ax2.lines[0]
        emissions = ax2.lines[1]
        lns = [baseline, emissions]
        labs = ['baseline', 'avg. emissions intensity']
        ax2.legend(lns, labs, fontsize=6, ncol=1, frameon=False, loc='lower left', bbox_to_anchor=(0.01, -0.03))
        ax22.legend(['revenue'], fontsize=6, ncol=1, frameon=False, loc='upper left', bbox_to_anchor=(0.01, 0.99))

        # Add text to denote sub-figures
        ax1.text(49, 1.02, 'a', fontsize=9, weight='bold')
        ax2.text(49, 1.02, 'b', fontsize=9, weight='bold')
        ax3.text(49, 1.02, 'c', fontsize=9, weight='bold')
        ax5.text(49, 1.02, 'e', fontsize=9, weight='bold')
        ax4.text(49, 1.02, 'd', fontsize=9, weight='bold')
        ax6.text(49, 1.02, 'f', fontsize=9, weight='bold')

        # Set figure size
        fig.set_size_inches(self.cm_to_in(14), self.cm_to_in(14) / 1.7)

        # Adjust subplot position
        fig.subplots_adjust(left=0.085, top=0.95, right=0.925, wspace=0.15)

        # Save figures
        fig.savefig(os.path.join(self.output_dir, 'calibration_intervals_comparison.png'), dpi=300)
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

        for c in ['1_calibration_intervals', '3_calibration_intervals', '6_calibration_intervals',
                  'revenue_target_1_ci', 'revenue_target_3_ci', 'revenue_target_6_ci']:
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

        # Line styles
        ci_1_style = {'color': self.colours['red'], 'alpha': 0.7, 'linewidth': 1}
        ci_3_style = {'color': self.colours['blue'], 'alpha': 0.7, 'linewidth': 1}
        ci_6_style = {'color': self.colours['green'], 'alpha': 0.7, 'linewidth': 1}
        emissions_style = {'color': self.colours['purple'], 'alpha': 0.7, 'linewidth': 0.7, 'linestyle': '--'}
        revenue_style = {'color': 'grey', 'alpha': 0.7, 'linewidth': 0.7, 'linestyle': '--'}

        # Plot revenue neutral baselines with different calibration intervals
        x = range(1, 53)
        ax1.plot(x, r['1_calibration_intervals']['baselines'].values, **ci_1_style)
        ax1.plot(x, r['3_calibration_intervals']['baselines'].values, **ci_3_style)
        ax1.plot(x, r['6_calibration_intervals']['baselines'].values, **ci_6_style)
        ax1.plot(x, r['1_calibration_intervals']['emissions_intensity'].values, **emissions_style)

        # Plot revenue targeting baselines
        ax2.plot(x, r['revenue_target_1_ci']['baselines'].values, **ci_1_style)
        ax2.plot(x, r['revenue_target_3_ci']['baselines'].values, **ci_3_style)
        ax2.plot(x, r['revenue_target_6_ci']['baselines'].values, **ci_6_style)
        ax2.plot(x, r['revenue_target_1_ci']['emissions_intensity'].values, **emissions_style)

        # Plot scheme revenue with revenue neutral target
        ax3.plot(x, r['1_calibration_intervals']['revenue'].values, **ci_1_style)
        ax3.plot(x, r['3_calibration_intervals']['revenue'].values, **ci_3_style)
        ax3.plot(x, r['6_calibration_intervals']['revenue'].values, **ci_6_style)

        # Plot revenue when positive revenue target is implemented
        ax4.plot(x, r['revenue_target_1_ci']['revenue'].values, **ci_1_style)
        ax4.plot(x, r['revenue_target_3_ci']['revenue'].values, **ci_3_style)
        ax4.plot(x, r['revenue_target_6_ci']['revenue'].values, **ci_6_style)

        # Plot revenue targets
        ax3.plot(x, [r['1_calibration_intervals']['parameters']['revenue_target'][2018][i] for i in x], **revenue_style)
        ax4.plot(x, [r['revenue_target_1_ci']['parameters']['revenue_target'][2018][i] for i in x], **revenue_style)

        # Set axis limit
        ax1.set_ylim([0.96, 1.03])
        ax2.set_ylim([0.96, 1.03])

        ax3.set_ylim([-0.7e7, 35e6])
        ax4.set_ylim([-0.7e7, 35e6])

        # Shade region
        ax2.fill_between([10, 20], [0, 0], [2, 2], color='k', alpha=0.2, linewidth=0)
        ax4.fill_between([10, 20], [-1e7, -1e7], [3.5e7, 3.5e7], color='k', alpha=0.2, linewidth=0)

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
            ax.yaxis.set_major_locator(plt.MultipleLocator(1e7))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5e7))

        # Add legend
        lns = ax3.lines + [ax1.lines[-1], ax4.lines[-1]]
        labs = ['1 interval', '3 intervals', '6 intervals', 'revenue target', 'avg. emissions intensity']
        ax3.legend(lns, labs, frameon=False, fontsize=7, loc=2, bbox_to_anchor=(-0.008, 1))

        # Plot titles
        ax1.set_title('Revenue neutral', fontsize=8)
        ax2.set_title('Positive revenue target', fontsize=8)
        ax1.title.set_position((ax1.title.get_position()[0], ax1.title.get_position()[1] - 0.05))
        ax2.title.set_position((ax2.title.get_position()[0], ax2.title.get_position()[1] - 0.05))

        # Add axes labels
        ax3.set_xlabel('Week', fontsize=8)
        ax4.set_xlabel('Week', fontsize=8)

        ax1.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=8)
        ax3.set_ylabel('Revenue ($)', fontsize=8)

        # Add text to denote sub-figures
        ax1.text(50.8, 0.965, 'a', fontsize=9, weight='bold')
        ax2.text(50.8, 0.965, 'b', fontsize=9, weight='bold')
        ax3.text(50.8, -0.45e7, 'c', fontsize=9, weight='bold')
        ax4.text(50.8, -0.45e7, 'd', fontsize=9, weight='bold')

        # Change size of scientific notation
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax3.yaxis.get_offset_text().set_size(7)
        ax3.yaxis.get_offset_text().set(ha='right', va='center')

        # Set figure size
        fig.set_size_inches(self.cm_to_in(14), self.cm_to_in(14) / 1.7)

        # Adjust subplot position
        fig.subplots_adjust(left=0.10, top=0.95, bottom=0.12, right=0.99, wspace=0.08, hspace=0.18)

        # Save figures
        fig.savefig(os.path.join(self.output_dir, 'revenue_targeting.png'), dpi=300)
        fig.savefig(os.path.join(self.output_dir, 'revenue_targeting.pdf'))

        plt.show()

    def get_revenue_floor_plot_data(self, use_cache=False, save=True):
        """Get data for revenue lower bound plot"""

        filename = 'revenue_floor_plot_data.pickle'

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

    def plot_revenue_floor(self):
        """Construct plot comparing implementation of revenue floor"""

        # Get data
        try:
            r = self.get_revenue_floor_plot_data(use_cache=True)
        except Exception as e:
            print(e)
            r = self.get_revenue_floor_plot_data(use_cache=False, save=True)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)

        # Change tick label size
        for ax in [ax1, ax2]:
            ax.tick_params(labelsize=9)

        # Plot revenue and baseline
        x = range(1, len(r['revenue_floor']['revenue'].values) + 1)

        # Styles
        with_floor = {'color': '#5887ed', 'linewidth': 0.8, 'alpha': 0.8, 'marker': 'o', 'markersize': 2.5,
                      'markeredgewidth': 0.5, 'markerfacecolor': 'none'}
        without_floor = {'color': '#db141e', 'linewidth': 0.8, 'alpha': 0.8, 'marker': 'o', 'markersize': 2.5,
                         'markeredgewidth': 0.5, 'markerfacecolor': 'none'}

        ax1.plot(x, r['revenue_floor']['baselines'].values, **with_floor)
        ax1.plot(x, r['3_calibration_intervals']['baselines'].values, **without_floor)

        ax2.plot(x, r['revenue_floor']['revenue'].values, **with_floor)
        ax2.plot(x, r['3_calibration_intervals']['revenue'].values, **without_floor)

        # Plot revenue floor line
        ax2.plot([0, 52], [0, 0], linewidth=0.7, linestyle=':', alpha=0.6, color='k')

        # Change size of scientific notation
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax2.yaxis.get_offset_text().set_size(9)
        ax2.yaxis.get_offset_text().set(ha='right', va='center')

        # Set labels for axes
        ax1.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=10, labelpad=0)
        ax1.set_xlabel('Week', fontsize=10)

        ax2.set_ylabel('Revenue ($)', fontsize=10, labelpad=0)
        ax2.set_xlabel('Week', fontsize=10)

        # Add text to denote sub-figures
        ax1.text(48, 0.97, 'a', fontsize=10, weight='bold')
        ax2.text(48, -2.85e6, 'b', fontsize=10, weight='bold')

        # Set major and minor ticks
        ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))

        ax1.yaxis.set_major_locator(plt.MultipleLocator(0.01))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.005))

        ax2.yaxis.set_major_locator(plt.MultipleLocator(1e6))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.5e6))

        # Add legend
        ax1.legend(['with floor', 'without floor'], fontsize=8, bbox_to_anchor=[0.5, 0.15], loc='center', frameon=False)

        # Set figure size
        fig.set_size_inches(self.cm_to_in(14), self.cm_to_in(14) / 2)

        # Adjust subplots
        fig.subplots_adjust(left=0.1, top=0.93, bottom=0.15, right=0.98, hspace=0.3, wspace=0.25)

        # Save figure
        fig.savefig(os.path.join(self.output_dir, 'revenue_floor.png'), dpi=300)
        fig.savefig(os.path.join(self.output_dir, 'revenue_floor.pdf'))

        plt.show()

    def get_scheme_eligibility_plot_data(self, use_cache=False, save=True):
        """Compare scenarios when renewables are eligible and ineligible to receive credits / penalties"""

        filename = 'renewable_eligibility_plot_data.pickle'

        # Load cached file if specified and return (faster loading)
        if use_cache:
            results = self.load_cache(self.output_dir, filename)
            return results

        # Results container
        results = {}

        for c in ['renewables_eligibility', '3_calibration_intervals']:
            # Get baselines and cumulative scheme revenue
            baselines, revenue = self.results.get_baselines_and_revenue(c)

            # Get DUIDs for different categories of generators
            thermal_duids = self.results.data.get_thermal_unit_duids()
            wind_duids = self.results.data.get_wind_unit_duids()
            solar_duids = self.results.data.get_solar_unit_duids()
            hydro_duids = self.results.data.get_hydro_unit_duids()

            # Get generators subjected to scheme
            # TODO: Need to update this to extract generators based on MPC model parameters
            if c == '3_calibration_intervals':
                scheme_generators = thermal_duids
            elif c == 'renewables_eligibility':
                scheme_generators = thermal_duids.union(wind_duids).union(solar_duids)
            else:
                raise Exception(f'Unexpected case: {c}')

            # Weekly prices
            nem_price, zone_price = self.results.get_week_prices(c)

            # Combine results
            results[c] = {'baselines': baselines, 'revenue': revenue,
                          'scheme_emissions_intensity': self.results.get_week_emissions_intensity(c, scheme_generators),
                          'system_emissions_intensity': self.results.get_week_system_emissions_intensity(c),
                          'cumulative_emissions': self.results.get_week_cumulative_emissions(c),
                          'nem_price': nem_price, 'zone_price': zone_price}

        # BAU prices
        bau_nem_price, bau_zone_price = self.results.get_week_prices('bau')

        # Get BAU emissions
        results['bau'] = {'cumulative_emissions': self.results.get_week_cumulative_emissions('bau'),
                          'system_emissions_intensity': self.results.get_week_system_emissions_intensity('bau'),
                          'nem_price': bau_nem_price, 'zone_price': bau_zone_price,
                          }

        # Save results
        if save:
            self.save(results, self.output_dir, filename)

        return results

    def plot_scheme_eligibility(self):
        """Compare baselines, scheme revenue, prices, and average emissions intensity when including renewables"""

        # Get data
        try:
            r = self.get_scheme_eligibility_plot_data(use_cache=True)
        except Exception as e:
            print(e)
            r = self.get_scheme_eligibility_plot_data(use_cache=False, save=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
        x = range(1, len(r['3_calibration_intervals']['baselines']) + 1)

        ineligible_style = {'color': self.colours['red'], 'alpha': 0.7, 'linewidth': 0.9, 'marker': 'o',
                            'markersize': 2.5, 'markeredgewidth': 0.5, 'markerfacecolor': 'none'}
        eligible_style = {'color': self.colours['blue'], 'alpha': 0.7, 'linewidth': 0.9, 'marker': 'o',
                          'markersize': 2.5, 'markeredgewidth': 0.5, 'markerfacecolor': 'none'}
        bau_style = {'color': self.colours['green'], 'alpha': 0.7, 'linewidth': 0.9, 'linestyle': '--',
                     'marker': 'o', 'markersize': 2.5, 'markeredgewidth': 0.5, 'markerfacecolor': 'none'}

        ax1.plot(x, r['3_calibration_intervals']['baselines'].values, **ineligible_style)
        ax1.plot(x, r['renewables_eligibility']['baselines'].values, **eligible_style)

        ax2.plot(x, r['3_calibration_intervals']['nem_price'].values, **ineligible_style)
        ax2.plot(x, r['renewables_eligibility']['nem_price'].values, **eligible_style)

        ax4.plot(x, r['3_calibration_intervals']['revenue'].values, **ineligible_style)
        ax4.plot(x, r['renewables_eligibility']['revenue'].values, **eligible_style)

        ax3.plot(x, r['3_calibration_intervals']['system_emissions_intensity'].values, **ineligible_style)
        ax3.plot(x, r['renewables_eligibility']['system_emissions_intensity'].values, **eligible_style)
        ax3.plot(x, r['bau']['system_emissions_intensity'].values, **bau_style)

        # Change size of scientific notation
        ax4.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax4.yaxis.get_offset_text().set_size(7)
        ax4.yaxis.get_offset_text().set(ha='right', va='center')

        # Change tick label size
        for ax in [ax1, ax2, ax4, ax3]:
            ax.tick_params(labelsize=7)

        # Set major and minor ticks
        ax1.yaxis.set_major_locator(plt.MultipleLocator(0.04))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.02))
        ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))

        ax2.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(5))

        ax4.yaxis.set_major_locator(plt.MultipleLocator(2e6))
        ax4.yaxis.set_minor_locator(plt.MultipleLocator(2e6 / 2))

        ax3.yaxis.set_major_locator(plt.MultipleLocator(0.04))
        ax3.yaxis.set_minor_locator(plt.MultipleLocator(0.02))

        # Include axes labels
        ax1.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=8, labelpad=-0.1)
        ax2.set_ylabel('Price ($/MWh)', fontsize=8, labelpad=-0.1)
        ax4.set_ylabel('Revenue ($)', fontsize=8, labelpad=-0.1)
        ax3.set_ylabel('Avg. emissions intensity \n (tCO$_{2}$/MWh)', fontsize=8, labelpad=-0.1)

        ax4.set_xlabel('Week', fontsize=8)
        ax3.set_xlabel('Week', fontsize=8)

        # Add text to denote sub-figures
        text_style = {'horizontalalignment': 'center', 'verticalalignment': 'center', 'fontsize': 9, 'weight': 'bold'}
        ax1.text(0.05, 0.08, 'a', transform=ax1.transAxes, **text_style)
        ax2.text(0.05, 0.08, 'b', transform=ax2.transAxes, **text_style)
        ax3.text(0.05, 0.08, 'c', transform=ax3.transAxes, **text_style)
        ax4.text(0.05, 0.08, 'd', transform=ax4.transAxes, **text_style)

        # Add legend
        lns = ax2.lines
        lns.reverse()
        labs = ['eligible', 'ineligible']
        ax2.legend(lns, labs, frameon=False, fontsize=7, loc='upper right',
                   # bbox_to_anchor=[0.8, 0.8],
                   )

        # Set axis for selected subplots
        ax1.set_ylim([0.89, 1.03])
        ax3.set_ylim([0.78, 0.96])

        # Add BAU emissions intensity legend
        ax3.legend([ax3.lines[-1]], ['BAU avg. emissions intensity'], frameon=False, fontsize=7, loc='center',
                   bbox_to_anchor=[0.5, 0.94])

        # Set figure size
        # fig.set_size_inches((self.cm_to_in(7.8), self.cm_to_in(6)))
        fig.set_size_inches(self.cm_to_in(14), self.cm_to_in(14) / 1.7)

        # Adjust subplot position
        fig.subplots_adjust(left=0.11, top=0.99, bottom=0.12, right=0.99, wspace=0.2, hspace=0.15)

        # Save figures
        fig.savefig(os.path.join(self.output_dir, 'renewables_eligibility.png'), dpi=300)
        fig.savefig(os.path.join(self.output_dir, 'renewables_eligibility.pdf'))

        plt.show()

    def get_historic_energy_output_data(self, use_cache=False, save=True):
        """Get data for a persistence based forecast plot"""

        filename = 'historic_energy_output_data.pickle'

        # Load cached file if specified and return (faster loading)
        if use_cache:
            results = self.load_cache(self.output_dir, filename)
            return results

        def get_data():
            """Connect to database and extract dispatch data for 2018"""

            # Connect to database and create cursor
            conn = MySQLdb.connect(host=os.environ['LOCAL_HOST'], user=os.environ['LOCAL_USER'],
                                   passwd=os.environ['LOCAL_PASSWORD'], db=os.environ['LOCAL_SCHEMA'],
                                   cursorclass=MySQLdb.cursors.DictCursor)
            cur = conn.cursor()

            # Run query and fetch results
            sql = "SELECT * FROM dispatch_unit_scada WHERE SETTLEMENTDATE >= '2018-01-01 00:05:00' AND SETTLEMENTDATE <= '2019-01-01 00:00:00'"
            cur.execute(sql)
            r = cur.fetchall()
            cur.close()
            conn.close()

            return r

        def parse_data(data):
            """Parse dispatch data from database. Aggregate to weekly resolution for each DUID."""

            # Convert to datetime objects
            df = pd.DataFrame(data)
            df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

            # Drop duplicates
            df = df.drop_duplicates(subset=['SETTLEMENTDATE', 'DUID'], keep='last')

            # Pivot
            df_d = df.pivot(index='SETTLEMENTDATE', columns='DUID', values='SCADAVALUE')

            # Convert dispatch interval power to energy
            df_d = df_d.astype(float)
            df_d = df_d.mul(15 / 60)

            # Re-sample to daily resolution. Set negative values to zero.
            df_w = df_d.resample('1w').sum()
            df_w[df_w < 0] = 0

            # Aggregate to station level
            df_s = df_w.T.join(self.results.data.generators[['STATIONID']]).groupby('STATIONID').sum().T

            # Drop the last row (incomplete)
            df_s = df_s.iloc[:-1]

            return df_s

        # Dispatch data from database
        dispatch_data = get_data()

        # Weekly energy
        energy = parse_data(dispatch_data)

        # Save data
        if save:
            with open(os.path.join(self.output_dir, filename), 'wb') as f:
                pickle.dump(energy, f)

        return energy

    def plot_forecast_comparison(self):
        """Plot persistence-based forecasting methodology justification and example"""

        # Persistence-based forecast
        r = self.get_historic_energy_output_data(use_cache=True)

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)

        # Colours for each generator
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(r.columns)))

        # Compute rolling mean
        r_mean = r.rolling(window=1000, min_periods=1).mean()

        # Plotting energy output in the current week with lagged output
        for lag, ax in [(-1, ax1), (-3, ax3), (-6, ax5)]:

            # Scatter plot for each station
            for i, duid in enumerate(r.columns):
                ax.scatter(r[duid].values, r[duid].shift(lag).values, alpha=0.2, s=2, color=colors[i], linewidths=0)

            # Use log-log axes
            ax.set_yscale('log')
            ax.set_xscale('log')

            # Set axes limits
            ax.set_ylim([100, 2e6])
            ax.set_xlim([100, 2e6])

            # Add line with slope 1
            ax.plot([100, 2e6], [100, 2e6], color='k', linestyle='--', alpha=0.7, linewidth=0.7)

        # Plotting energy output in the current week with lagged output
        for lag, ax in [(-1, ax2), (-3, ax4), (-6, ax6)]:

            # Scatter plot for each station
            for i, duid in enumerate(r.columns):
                ax.scatter(r_mean[duid].values, r[duid].shift(lag).values, alpha=0.2, s=2, color=colors[i],
                           linewidths=0)

            # Use log-log axes
            ax.set_yscale('log')
            ax.set_xscale('log')

            # Set axes limits
            ax.set_ylim([100, 2e6])
            ax.set_xlim([100, 2e6])

            # Add line with slope 1
            ax.plot([100, 2e6], [100, 2e6], color='k', linestyle='--', alpha=0.7, linewidth=0.7)

        # Axes labels
        label_properties = {'fontsize': 9}
        ax1.set_ylabel('Energy +1 week (MWh)', **label_properties)
        ax2.set_ylabel('Energy +1 week (MWh)', **label_properties)
        ax3.set_ylabel('Energy +3 weeks (MWh)', **label_properties)
        ax4.set_ylabel('Energy +3 weeks (MWh)', **label_properties)
        ax5.set_ylabel('Energy +6 weeks (MWh)', **label_properties)
        ax6.set_ylabel('Energy +6 weeks (MWh)', **label_properties)

        ax5.set_xlabel('Energy (MWh)', **label_properties)
        ax6.set_xlabel('Energy (MWh)', **label_properties)

        # Change tick label size
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.tick_params(axis='both', labelsize=8)

        # Set major and minor ticks
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

            ax.yaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        # Add text to denote sub-figures
        ax1.text(0.6e6, 180, 'a', fontsize=9, weight='bold')
        ax2.text(0.6e6, 180, 'b', fontsize=9, weight='bold')
        ax3.text(0.6e6, 180, 'c', fontsize=9, weight='bold')
        ax4.text(0.6e6, 180, 'd', fontsize=9, weight='bold')
        ax5.text(0.6e6, 180, 'e', fontsize=9, weight='bold')
        ax6.text(0.6e6, 180, 'f', fontsize=9, weight='bold')

        # Add title to persistence and mean-based forecasts
        ax1.set_title('Persistence-based', fontsize=9, pad=3)
        ax2.set_title('Climatology-based', fontsize=9, pad=3)

        # Set figure size
        fig.set_size_inches(self.cm_to_in(14), self.cm_to_in(14) * 1.3)

        # Adjust subplots
        fig.subplots_adjust(left=0.10, top=0.97, bottom=0.06, right=0.98, hspace=0.15, wspace=0.3)

        # Save figure
        fig.savefig(os.path.join(self.output_dir, 'forecast_comparison.png'), dpi=300)
        fig.savefig(os.path.join(self.output_dir, 'forecast_comparison.pdf'))

        plt.show()

    def get_multi_scenario_input_plot_data(self, use_cache=False, save=True):
        """Get data used to show construction procedure for multi-scenario generation procedure"""

        filename = 'multi_scenario_input_plot_data.pickle'

        # Load cached file if specified and return (faster loading)
        if use_cache:
            results = self.load_cache(self.output_dir, filename)
            return results

        # Multi-scenario generation input plot
        energy = self.results.get_generator_energy('multi_scenario_forecast')

        # Arrange DataFrame so format is suitable for forecast generation
        df_e = energy.reset_index().pivot_table(index=['year', 'week', 'interval'], columns='generator',
                                                values='energy')

        # Group-by year and week
        df_e_wk = df_e.groupby(['year', 'week']).sum()

        # Filter so only have 2017 values
        df_e_wk_f = df_e_wk.loc[df_e_wk.index.get_level_values(0) == 2017, :]

        # Object used to construct forecasts
        forecast = MonteCarloForecast()

        # Generate forecasts
        duid = 'ER02'
        scenario_energy, scenario_weights, energy_paths = forecast.get_duid_scenarios(df_e_wk_f, duid, n_intervals=3,
                                                                                      n_random_paths=500, n_clusters=5)

        # Combine results into single dictionary
        results = {'observed_energy': df_e_wk_f, 'scenario_energy': scenario_energy, 'energy_paths': energy_paths,
                   'duid': duid}

        # Save results
        if save:
            self.save(results, self.output_dir, filename)

        return results

    def plot_multi_scenario_input(self):
        """Plot showing underlying series and histogram of first difference of underlying"""

        # Get data
        try:
            r = self.get_multi_scenario_input_plot_data(use_cache=True)
        except Exception as e:
            print(e)
            r = self.get_multi_scenario_input_plot_data(use_cache=False, save=True)

        # Create plots
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)

        # Generator for which to construct example
        duid = r['duid']

        # Plot observed energy
        x_obs = range(1, r['observed_energy'].shape[0] + 1)
        ax1.plot(x_obs, r['observed_energy'][duid].values, color='#029e4d', linewidth=0.8)

        # Plot observed data
        ax2.plot(range(48, 53), r['observed_energy'].iloc[-5:][duid].values, color='#029e4d', linewidth=0.8)

        # Plot random paths
        for p in r['energy_paths']:
            x_forecast = range(r['observed_energy'].shape[0], r['observed_energy'].shape[0] + 3 + 1)
            ax2.plot(x_forecast, p, color='b', alpha=0.02, linewidth=0.8)

        # Plot centroids
        centroids = [[r['observed_energy'].iloc[-1][duid]] + [r['scenario_energy'][(duid, s, t)] for t in range(1, 4)]
                     for s in range(1, 6)]

        for c in centroids:
            x_forecast = range(r['observed_energy'].shape[0], r['observed_energy'].shape[0] + 3 + 1)
            ax2.plot(x_forecast, c, color='#e60202', alpha=0.9, linewidth=0.8)

        # Axes labels
        ax1.set_xlabel('Week', fontsize=7)
        ax2.set_xlabel('Week', fontsize=7)
        ax1.set_ylabel('Energy (MWh)', fontsize=7)

        # Change tick label size
        for ax in [ax1, ax2]:
            ax.tick_params(labelsize=7)

        # Format major and minor ticks
        ax1.yaxis.set_major_locator(plt.MultipleLocator(2000))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(1000))

        ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))

        ax2.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))

        # Add text to denote sub-figures
        ax1.text(2, 7300, 'a', fontsize=9, weight='bold')
        ax2.text(48.1, 7300, 'b', fontsize=9, weight='bold')

        # Add text to denote DUID under investigation
        ax2.text(53, 14750, duid, fontsize=7)

        # Set figure size
        fig.set_size_inches((self.cm_to_in(7.8), self.cm_to_in(5)))

        # Adjust subplots
        fig.subplots_adjust(left=0.20, bottom=0.19, top=0.99, right=0.99)

        # Save figure
        fig.savefig(os.path.join(self.output_dir, 'multi_scenario_input.png'))
        fig.savefig(os.path.join(self.output_dir, 'multi_scenario_input.pdf'))

        plt.show()

    def get_emissions_intensity_shock_plot_data(self, use_cache=False, save=True):
        """Get data showing baseline and revenue response following an emissions intensity shock to the system"""

        filename = 'emissions_intensity_shock_plot_data.pickle'

        # Load cached file if specified and return (faster loading)
        if use_cache:
            results = self.load_cache(self.output_dir, filename)
            return results

        # Container for results
        results = {}

        for c in ['anticipated_emissions_intensity_shock_1_ci', 'anticipated_emissions_intensity_shock_3_ci',
                  'anticipated_emissions_intensity_shock_6_ci', 'unanticipated_emissions_intensity_shock_1_ci',
                  'unanticipated_emissions_intensity_shock_3_ci', 'unanticipated_emissions_intensity_shock_6_ci']:
            # Generators under scheme's remit
            generators = self.results.data.get_thermal_unit_duids()

            # Extract baselines, scheme revenue, and emissions intensity of regulated generators
            baselines, revenue = self.results.get_baselines_and_revenue(c)
            emissions_intensity = self.results.get_week_emissions_intensity(c, generators)

            # Place results in dictionary
            results[c] = {'baselines': baselines, 'revenue': revenue, 'emissions_intensity': emissions_intensity}

        # Save results
        if save:
            self.save(results, self.output_dir, filename)

        return results

    def plot_emissions_intensity_shock(self):
        """Plot response of baseline and scheme revenue to a persistent change to generator emissions intensities"""

        # Get data
        r = self.get_emissions_intensity_shock_plot_data(use_cache=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)

        # Share selected axes
        ax1.get_shared_y_axes().join(ax1, ax2)
        ax1.get_shared_y_axes().join(ax3, ax4)

        # Plot anticipated shock baselines
        x = range(1, 53)
        ci_1_style = {'color': self.colours['red'], 'linewidth': 1.3, 'alpha': 0.8}
        ci_3_style = {'color': self.colours['blue'], 'linewidth': 1.3, 'alpha': 0.8}
        ci_6_style = {'color': self.colours['green'], 'linewidth': 1.3, 'alpha': 0.8}
        emissions_style = {'color': self.colours['purple'], 'linewidth': 0.8, 'linestyle': '--', 'alpha': 0.8}
        revenue_style = {'color': 'grey', 'linewidth': 0.8, 'linestyle': '--', 'alpha': 0.8}
        shock_style = {'color': 'k', 'linestyle': ':', 'linewidth': 0.8, 'alpha': 0.8}

        # Plot anticipated shock baselines
        ax1.plot(x, r['anticipated_emissions_intensity_shock_1_ci']['baselines'].values, **ci_1_style)
        ax1.plot(x, r['anticipated_emissions_intensity_shock_3_ci']['baselines'].values, **ci_3_style)
        ax1.plot(x, r['anticipated_emissions_intensity_shock_6_ci']['baselines'].values, **ci_6_style)

        # Plot unanticipated shock baselines
        ax2.plot(x, r['unanticipated_emissions_intensity_shock_1_ci']['baselines'].values, **ci_1_style)
        ax2.plot(x, r['unanticipated_emissions_intensity_shock_3_ci']['baselines'].values, **ci_3_style)
        ax2.plot(x, r['unanticipated_emissions_intensity_shock_6_ci']['baselines'].values, **ci_6_style)

        # Plot anticipated shock revenue
        ax3.plot(x, r['anticipated_emissions_intensity_shock_1_ci']['revenue'].values, **ci_1_style)
        ax3.plot(x, r['anticipated_emissions_intensity_shock_3_ci']['revenue'].values, **ci_3_style)
        ax3.plot(x, r['anticipated_emissions_intensity_shock_6_ci']['revenue'].values, **ci_6_style)

        # Plot unanticipated shock revenue
        ax4.plot(x, r['unanticipated_emissions_intensity_shock_1_ci']['revenue'].values, **ci_1_style)
        ax4.plot(x, r['unanticipated_emissions_intensity_shock_3_ci']['revenue'].values, **ci_3_style)
        ax4.plot(x, r['unanticipated_emissions_intensity_shock_6_ci']['revenue'].values, **ci_6_style)

        # Plot system emissions intensity
        ax1.plot(x, r['anticipated_emissions_intensity_shock_1_ci']['emissions_intensity'].values, **emissions_style)
        ax2.plot(x, r['unanticipated_emissions_intensity_shock_1_ci']['emissions_intensity'].values, **emissions_style)

        # Plot revenue target
        ln_rev = ax3.plot([-1, 55], [0, 0], **revenue_style)
        ax4.plot([-1, 55], [0, 0], **revenue_style)

        # Add line denoting week of shock
        ax1.plot([10, 10], [0.6, 1.1], **shock_style)
        ax2.plot([10, 10], [0.6, 1.1], **shock_style)
        ax3.plot([10, 10], [-4e7, 4e7], **shock_style)
        ax4.plot([10, 10], [-4e7, 4e7], **shock_style)

        # Set axes limits
        ax1.set_ylim([0.68, 1.05])
        ax1.set_xlim([0, 53])
        ax2.set_ylim([0.68, 1.05])
        ax3.set_ylim([-3.8e7, 3.5e7])
        ax4.set_ylim([-3.8e7, 3.5e7])

        # Add legend
        lns = ax2.lines[:-1] + ln_rev
        labs = ['1 interval', '3 intervals', '6 intervals', 'avg. emissions intensity', 'revenue target']
        ax2.legend(lns, labs, fontsize=6, loc='upper right', frameon=False)

        # Add axes labels and plot titles
        ax3.set_xlabel('Week', fontsize=9)
        ax4.set_xlabel('Week', fontsize=9)
        ax1.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=9)
        ax3.set_ylabel('Revenue ($)', fontsize=9)
        ax1.set_title('Anticipated', fontsize=8, position=(0.5, 0.97))
        ax2.set_title('Unanticipated', fontsize=8, position=(0.5, 0.97))

        # Add text to denote sub-figures
        ax1.text(49.15, 0.7, 'a', fontsize=9, weight='bold')
        ax2.text(49.15, 0.7, 'b', fontsize=9, weight='bold')
        ax3.text(49.15, -3.4e7, 'c', fontsize=9, weight='bold')
        ax4.text(49.15, -3.4e7, 'd', fontsize=9, weight='bold')

        # Format major and minor ticks
        ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))

        ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

        ax3.yaxis.set_major_locator(plt.MultipleLocator(2e7))
        ax3.yaxis.set_minor_locator(plt.MultipleLocator(1e7))
        ax4.yaxis.set_major_locator(plt.MultipleLocator(2e7))
        ax4.yaxis.set_minor_locator(plt.MultipleLocator(1e7))

        # Use scientific notation for revenue
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax3.yaxis.get_offset_text().set_size(7)
        ax3.yaxis.get_offset_text().set(ha='right', va='center')

        # Change fontsize for tick labels
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(labelsize=7)

        # Remove tick labels
        ax2.axes.yaxis.set_ticklabels([])
        ax4.axes.yaxis.set_ticklabels([])

        # Set figure size
        fig.set_size_inches(self.cm_to_in(14), self.cm_to_in(14) / 1.6)

        # Adjust subplot position
        fig.subplots_adjust(left=0.09, top=0.96, bottom=0.11, right=0.99, wspace=0.07, hspace=0.15)

        # Save figures
        fig.savefig(os.path.join(self.output_dir, 'emissions_intensity_shock.png'), dpi=300)
        fig.savefig(os.path.join(self.output_dir, 'emissions_intensity_shock.pdf'))

        plt.show()

    def get_multi_scenario_plot_data(self, use_cache=False, save=True):
        """Get data comparing persistence-based forecast with multi-scenario forecasting strategy"""

        filename = 'multi_scenario_plot_data.pickle'

        # Load cached file if specified and return (faster loading)
        if use_cache:
            results = self.load_cache(self.output_dir, filename)
            return results

        # Container for results
        results = {}

        for c in ['persistence_forecast', 'multi_scenario_forecast']:
            # Extract baselines and scheme revenue
            baselines, revenue = plots.results.get_baselines_and_revenue(c)

            # Append to results container
            results[c] = {'baselines': baselines, 'revenue': revenue}

        # Save results
        if save:
            self.save(results, self.output_dir, filename)

        return results

    def plot_calibration_interval_baseline_revenue_variability(self):
        """Show how number of calibration intervals impacts scheme revenue and baseline variance"""

        # Get data
        r = self.get_calibration_interval_comparison_plot_data(use_cache=True)

        # Calibration intervals
        x = range(1, 7)

        # Standard deviation of baselines and scheme revenue
        baselines = [r[i]['baselines'].std() for i in range(1, 7)]
        revenue = [r[i]['revenue'].std() for i in range(1, 7)]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        line_properties = {'linewidth': 0.9, 'linestyle': '--', 'marker': 'o', 'markersize': 4,
                           'markeredgewidth': 0.6, 'markerfacecolor': 'none'}
        ln1 = ax1.plot(x, baselines, color='#3482ba', **line_properties)

        ln2 = ax2.plot(x, revenue, color='#ba2323', **line_properties)

        # Set axes labels
        ax1.set_ylabel('Baseline SD (tCO$_{2}$/MWh)')
        ax2.set_ylabel('Revenue SD ($)')
        ax1.set_xlabel('Calibration intervals')

        # Set font size
        ax1.xaxis.label.set_size(10)
        ax1.yaxis.label.set_size(10)

        ax2.yaxis.label.set_size(10)

        # Change size of scientific notation
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax2.yaxis.get_offset_text().set_size(9)
        ax2.yaxis.get_offset_text().set(ha='left', va='center')

        # Set major and minor ticks
        ax1.yaxis.set_major_locator(plt.MultipleLocator(0.002))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.0005))

        ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2e6))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.05e6))

        # Change tick label size
        for ax in [ax1, ax2]:
            ax.tick_params(labelsize=9)

        # Add legend
        lns = ln1 + ln2
        ax1.legend(lns, ['baseline SD', 'revenue SD'], fontsize=9, loc='upper center', frameon=False,
                   bbox_to_anchor=[0.4, 1])

        # Set figure size
        fig.set_size_inches(self.cm_to_in(14), self.cm_to_in(14) / 1.7)

        # Adjust subplots
        fig.subplots_adjust(left=0.13, top=0.95, bottom=0.13, right=0.9)

        # Save figure
        fig.savefig(os.path.join(self.output_dir, 'baseline_revenue_variability_comparison.png'), dpi=300)
        fig.savefig(os.path.join(self.output_dir, 'baseline_revenue_variability_comparison.pdf'))

        plt.show()

    def get_carbon_tax_rep_scheme_comparison_plot_data(self, use_cache=False, save=True):
        """Get data used to compare REP scheme with a carbon tax"""

        filename = 'carbon_tax_rep_scheme_comparison_plot_data.pickle'

        # Load cached file if specified and return (faster loading)
        if use_cache:
            results = self.load_cache(self.output_dir, filename)
            return results

        # Container for results
        results = {}

        for c in ['bau', '3_calibration_intervals']:
            # Extract baselines, scheme revenue, and prices
            baselines, revenue = self.results.get_baselines_and_revenue(c)
            prices, _ = self.results.get_week_prices(c)

            # Place results in dictionary
            results[c] = {'baselines': baselines, 'revenue': revenue, 'prices': prices}

        # Energy output from generators under a carbon tax
        energy = self.results.get_generator_energy('carbon_tax')
        df = energy.reset_index().pivot_table(index=['year', 'week', 'day', 'interval'], columns='generator',
                                              values='energy')

        # Total emissions from each generator
        emissions = df.apply(lambda x: x * self.results.data.generators['EMISSIONS'], axis=1)

        # Tax paid by each generator in each calibration interval
        # TODO: May want to generalise extraction of permit price. OK for now because 40 $/tCO2 used in all cases.
        tax = emissions.mul(40)

        # Tax revenue collected each week
        weekly_tax_revenue = tax.groupby(['year', 'week']).sum().sum(axis=1)

        # Get prices under a carbon tax
        prices_tax, _ = self.results.get_week_prices('carbon_tax')

        # Add results to dictionary to save
        results['carbon_tax'] = {'prices': prices_tax, 'revenue': weekly_tax_revenue}

        # Save results
        if save:
            self.save(results, self.output_dir, filename)

        return results

    def plot_carbon_tax_rep_scheme_comparison(self):
        """Compare price and revenue outcomes between a REP and carbon tax"""

        # Compare REP scheme with carbon tax
        r = self.get_carbon_tax_rep_scheme_comparison_plot_data(use_cache=True)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        x1 = range(1, 53)
        line_properties = {'linewidth': 0.8, 'alpha': 0.8, 'marker': 'o', 'markersize': 2.5, 'markeredgewidth': 0.6,
                           'markerfacecolor': 'none'}
        ax1.plot(x1, r['bau']['prices'].values, color='#8731bd', **line_properties)
        ax1.plot(x1, r['3_calibration_intervals']['prices'].values, color='#4584f7', **line_properties)
        ax1.plot(x1, r['carbon_tax']['prices'].values, color='#c4232b', **line_properties)

        x2 = range(0, 53)
        ax2.plot(x2, [0] + list(r['3_calibration_intervals']['revenue'].values), color='#4584f7', **line_properties)
        ax2.plot(x2, [0] + list(r['carbon_tax']['revenue'].cumsum().values), color='#c4232b', **line_properties)

        # Axes labels
        label_properties = {'fontsize': 9}
        ax1.set_ylabel('Price ($/MWh)', **label_properties)
        ax2.set_ylabel('Revenue ($)', **label_properties)

        ax1.set_xlabel('Week', **label_properties)
        ax2.set_xlabel('Week', **label_properties)

        # Change tick label size
        for ax in [ax1, ax2]:
            ax.tick_params(labelsize=8)

        # Set major and minor ticks
        ax1.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(5))

        ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))

        ax2.yaxis.set_major_locator(plt.MultipleLocator(1e9))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.5e9))

        ax2.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax2.xaxis.set_minor_locator(plt.MultipleLocator(5))

        # Change size of scientific notation
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax2.yaxis.get_offset_text().set_size(8)
        ax2.yaxis.get_offset_text().set(ha='center')

        # Add text to denote sub-figures
        ax1.text(0.92, 0.92, 'a', fontsize=9, weight='bold', transform=ax1.transAxes)
        ax2.text(0.05, 0.92, 'b', fontsize=9, weight='bold', transform=ax2.transAxes)

        # Add legend
        ax1.legend(['BAU', 'REP', 'Tax'], fontsize=8, loc='center right', frameon=False, bbox_to_anchor=[1, 0.45])

        # Set figure size
        fig.set_size_inches(self.cm_to_in(14), self.cm_to_in(14) / 1.9)

        # Adjust subplots
        fig.subplots_adjust(left=0.08, top=0.94, bottom=0.15, right=0.99, hspace=0.3, wspace=0.2)

        # Save figure
        fig.savefig(os.path.join(self.output_dir, 'rep_vs_carbon_tax.png'), dpi=300)
        fig.savefig(os.path.join(self.output_dir, 'rep_vs_carbon_tax.pdf'))

        plt.show()


if __name__ == '__main__':
    figures_output_directory = os.path.join(os.path.dirname(__file__), 'output', 'figures', 'thesis')

    # Objects used to create plots and tables
    plots = CreatePlots(figures_output_directory)

    # Plot baseline and revenue for different calibration interval durations
    r1 = plots.get_calibration_interval_comparison_plot_data(use_cache=False)
    plots.plot_calibration_interval_comparison()

    # Revenue targeting plot
    r2 = plots.get_revenue_targeting_plot_data(use_cache=False)
    plots.plot_revenue_targeting()

    # Revenue floor plot
    r3 = plots.get_revenue_floor_plot_data(use_cache=False)
    plots.plot_revenue_floor()

    # Scheme eligibility
    r4 = plots.get_scheme_eligibility_plot_data(use_cache=False)
    plots.plot_scheme_eligibility()

    # Emissions intensity shock
    r5 = plots.get_emissions_intensity_shock_plot_data(use_cache=False)
    plots.plot_emissions_intensity_shock()

    # Persistence-based forecast
    # NOTE: running this assumes MMSDM DISPATCHUNITSCADA data has been downloaded and stored within a MySQL database
    r6 = plots.get_historic_energy_output_data(use_cache=True)
    plots.plot_forecast_comparison()

    # Compare variability associated with baseline and scheme revenue for different calibration intervals
    r7 = plots.get_calibration_interval_comparison_plot_data(use_cache=False)
    plots.plot_calibration_interval_baseline_revenue_variability()

    # Compare prices and scheme revenue under a carbon tax and REP scheme
    r8 = plots.get_carbon_tax_rep_scheme_comparison_plot_data(use_cache=False)
    plots.plot_carbon_tax_rep_scheme_comparison()

