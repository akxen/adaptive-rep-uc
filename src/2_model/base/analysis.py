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
        s = pd.Series(results[var_id])

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
    def get_cumulative_scheme_revenue(output_dir, year, week):
        """Compute cumulative scheme revenue. Net revenue at the start of 'week'."""

        files = [f for f in os.listdir(output_dir) if 'interval_' in f]

        cumulative_revenue = 0

        for f in files:
            y, w, d = int(f.split('_')[1]), int(f.split('_')[2]), int(f.split('_')[3].replace('.pickle', ''))

            if (y <= year) and (w < week):
                with open(os.path.join(output_dir, f), 'rb') as g:
                    results = pickle.load(g)

                # Update cumulative scheme revenue
                cumulative_revenue += results['DAY_SCHEME_REVENUE']

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
                interval_revenue[(year, week)] += results['DAY_SCHEME_REVENUE']
            else:
                interval_revenue[(year, week)] = results['DAY_SCHEME_REVENUE']

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
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output')

    # Class used to analysis and process model results
    analysis = AnalyseResults()

    # Scheme revenue
    cumulative_revenue = analysis.get_cumulative_scheme_revenue(output_directory, 2018, 2)

    # Cumulative scheme revenue for each week in model horizon
    model_revenue = {(y, w): analysis.get_cumulative_scheme_revenue(output_directory, y, w) for w in range(1, 53)
                     for y in [2018]}

    # Model baselines
    model_baselines = analysis.get_baselines(output_directory)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    df_revenue = pd.Series(model_revenue).sort_index().rename('cumulative_revenue')
    df_baselines = pd.Series(model_baselines).sort_index().rename('baseline')

    # Combine into single DataFrame
    df_c = pd.concat([df_revenue, df_baselines], axis=1)

    # Plot cumulative scheme revenue and baselines
    df_c['cumulative_revenue'].plot(ax=ax1, color='blue')
    df_c['baseline'].plot(ax=ax2, color='red')

    plt.show()

    # Cumulative scheme revenue at beginning of week 2
    cumulative_revenue_2 = analysis.get_cumulative_scheme_revenue(output_directory, 2018, 2)
    # cumulative_revenue_3 = analysis.get_cumulative_scheme_revenue(output_directory, 2018, 3)
    # cumulative_revenue_4 = analysis.get_cumulative_scheme_revenue(output_directory, 2018, 4)

    with open(os.path.join(output_directory, 'mpc_2018_2.pickle'), 'rb') as f:
        mpc_results_2 = pickle.load(f)

    with open(os.path.join(output_directory, 'mpc_2018_3.pickle'), 'rb') as f:
        mpc_results_3 = pickle.load(f)

    # with open(os.path.join(output_directory, 'mpc_2018_4.pickle'), 'rb') as f:
    #     mpc_results_4 = pickle.load(f)
