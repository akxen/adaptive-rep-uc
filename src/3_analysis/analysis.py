"""Analyse results"""

import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'base'))

import pandas as pd
import matplotlib.pyplot as plt


class AnalyseResults:
    def __init__(self):
        pass

    @staticmethod
    def load_interval_results(output_dir, year, week, day):
        """Load results for a given interval"""

        with open(os.path.join(output_dir, f'interval_{year}_{week}_{day}.pickle'), 'rb') as f:
            results = pickle.load(f)

        return results

    def get_generator_interval_results(self, output_dir, var_id, year, week, day):
        """Get results relating to a given generator. Index should be (generator, hour)."""

        # Load results
        results = self.load_interval_results(output_dir, year, week, day)

        # Convert to series
        s = pd.Series(results[(year, week, day)][var_id])

        # Rename axes
        df = s.rename_axis(['generator', 'hour']).rename(var_id)

        # Reset index, pivot and only take first 24 hours. Remaining sequence will overlap with next day.
        df = df.reset_index().pivot(index='hour', columns='generator', values=var_id).loc[1:24]

        # Add interval ID to results DataFrame
        df = pd.concat([df], keys=[(year, week, day)], names=['year', 'week', 'day'])

        return df

    @staticmethod
    def get_baselines(output_dir):
        """Get baselines over model horizon"""

        # Files containing MPC model information
        files = [f for f in os.listdir(output_dir) if 'mpc_' in f]

        # Container for baselines
        baselines = {}

        # Loop through model result files, extracting baselines
        for f in files:
            year, week = int(f.split('_')[1]), int(f.split('_')[2].replace('.pickle', ''))

            with open(os.path.join(output_dir, f), 'rb') as g:
                results = pickle.load(g)

            # Add baseline to container
            baselines[(year, week)] = results['baseline_trajectory'][1]

        return baselines

    @staticmethod
    def get_cumulative_scheme_revenue(output_dir):
        """Get baselines over model horizon"""

        files = [f for f in os.listdir(output_dir) if 'mpc_' in f]

        cumulative_revenue = {}

        for f in files:
            year, week = int(f.split('_')[1]), int(f.split('_')[2].replace('.pickle', ''))

            with open(os.path.join(output_dir, f), 'rb') as g:
                results = pickle.load(g)

            # Add baseline to container
            cumulative_revenue[(year, week)] = results['parameters']['revenue_start']

        return cumulative_revenue

    def get_interval_scheme_revenue(self, output_dir):
        """Get baselines over model horizon"""

        files = [f for f in os.listdir(output_dir) if 'interval_' in f]

        interval_revenue = {}

        for f in files:
            year, week, day = int(f.split('_')[1]), int(f.split('_')[2]), int(f.split('_')[3].replace('.pickle', ''))

            # Results for a given interval
            results = self.load_interval_results(output_dir, year, week, day)

            # Add interval to container
            if (year, week) in interval_revenue.keys():
                interval_revenue[(year, week)] += results[(year, week, day)]['DAY_SCHEME_REVENUE']
            else:
                interval_revenue[(year, week)] = results[(year, week, day)]['DAY_SCHEME_REVENUE']

        return interval_revenue

    def get_generator_energy_output(self, output_dir, year, week):
        """Compute total generator energy output for a given week"""

        # Aggregate energy output for each week
        df_e = pd.concat([self.get_generator_interval_results(output_dir, 'e', year, week, d) for d in range(1, 8)])

        # Transform into a DataFrame with aggregate energy output for each generator in a single column
        df_o = (df_e.groupby(['year', 'week']).sum().reset_index().drop(['year', 'week'], axis=1).T
                .rename(columns={0: 'energy_observed'}))

        return df_o


if __name__ == '__main__':
    # Directory containing output files
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'output')

    # Class used to analysis and process model results
    analysis = AnalyseResults()

    # Baselines, cumulative scheme revenue, and revenue obtained each interval
    model_baselines = analysis.get_baselines(output_directory)
    model_cumulative_revenue = analysis.get_cumulative_scheme_revenue(output_directory)
    model_interval_revenue = analysis.get_interval_scheme_revenue(output_directory)

    # Convert to DataFrames
    df_b = pd.Series(model_baselines)
    df_r = pd.Series(model_cumulative_revenue)
    df_i = pd.Series(model_interval_revenue)

    # Plotting to check data
    df_b.sort_index().plot()
    plt.show()
    df_r.sort_index().plot()
    plt.show()
    df_i.sort_index().plot()
    plt.show()

    filename = 'mpc_2018_3.pickle'
    with open(os.path.join(output_directory, filename), 'rb') as g:
        interval_results = pickle.load(g)

    df_e_actual = analysis.get_generator_energy_output(output_directory, 2018, 3)

    df_e_forecast = (pd.Series(interval_results['energy_forecast']).rename('energy_forecast')
        .rename_axis(['generator', 'interval']).reset_index().drop('interval', axis=1).set_index('generator'))

    df_e_compare = pd.concat([df_e_actual, df_e_forecast], axis=1, sort=True)

    fig, ax = plt.subplots()
    df_e_compare.plot.scatter(x='energy_observed', y='energy_forecast', ax=ax)
    ax.plot([0, 120000], [0, 120000])
    # for k, v in df_e_compare.iterrows():
    #     ax.annotate(k, v)
    plt.show()
