"""Analyse results"""

import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'base'))

import pandas as pd
import matplotlib.pyplot as plt

from data import ModelData
from mpc import MPCController


def get_baselines(output_dir):
    """Get baselines over model horizon"""

    files = [f for f in os.listdir(output_dir) if 'mpc_' in f]

    baselines = {}

    for f in files:
        year, week = int(f.split('_')[1]), int(f.split('_')[2].replace('.pickle', ''))

        with open(os.path.join(output_dir, f), 'rb') as g:
            results = pickle.load(g)

        # Add baseline to container
        baselines[(year, week)] = results['baseline_trajectory'][1]

    return baselines


def get_cumulative_scheme_revenue(output_dir):
    """Get baselines over model horizon"""

    files = [f for f in os.listdir(output_dir) if 'mpc_' in f]

    cumulative_revenue = {}

    for f in files:
        year, week = int(f.split('_')[1]), int(f.split('_')[2].replace('.pickle', ''))

        with open(os.path.join(output_dir, f), 'rb') as g:
            results = pickle.load(g)

        # Add baseline to container
        # cumulative_revenue[(year, week)] = results['cumulative_revenue'][1] + results['parameters']['revenue_start']
        cumulative_revenue[(year, week)] = results['parameters']['revenue_start']

    return cumulative_revenue


def get_interval_scheme_revenue(output_dir):
    """Get baselines over model horizon"""

    files = [f for f in os.listdir(output_dir) if 'interval_' in f]

    interval_revenue = {}

    for f in files:
        year, week, day = int(f.split('_')[1]), int(f.split('_')[2]), int(f.split('_')[3].replace('.pickle', ''))

        with open(os.path.join(output_dir, f), 'rb') as g:
            results = pickle.load(g)

        # Add interval to container
        if (year, week) in interval_revenue.keys():
            interval_revenue[(year, week)] += results[(year, week, day)]['DAY_SCHEME_REVENUE']
        else:
            interval_revenue[(year, week)] = results[(year, week, day)]['DAY_SCHEME_REVENUE']

    return interval_revenue


def get_generator_energy_output(output_dir, year, week):
    """Compute total generator energy output for a given week"""

    # Object used to parse generator results
    mpc_obj = MPCController(output_dir)

    # Aggregate energy output for each week
    df_e = pd.concat([mpc_obj.get_generator_interval_results('e', year, week, d) for d in range(1, 8)])

    # Transform into a DataFrame with aggregate energy output for each generator in a single column
    df_o = (df_e.groupby(['year', 'week']).sum().reset_index().drop(['year', 'week'], axis=1).T
            .rename(columns={0: 'energy_observed'}))

    return df_o


if __name__ == '__main__':
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'output')

    mpc = MPCController(output_directory)

    model_baselines = get_baselines(output_directory)
    model_cumulative_revenue = get_cumulative_scheme_revenue(output_directory)
    model_interval_revenue = get_interval_scheme_revenue(output_directory)

    df_b = pd.Series(model_baselines)
    df_r = pd.Series(model_cumulative_revenue)
    df_i = pd.Series(model_interval_revenue)

    df_b.sort_index().plot()
    plt.show()
    df_r.sort_index().plot()
    plt.show()
    df_i.sort_index().plot()
    plt.show()

    f = 'mpc_2018_3.pickle'
    with open(os.path.join(output_directory, f), 'rb') as g:
        results = pickle.load(g)

    df_e_actual = get_generator_energy_output(output_directory, 2018, 3)

    df_e_forecast = (pd.Series(results['energy_forecast']).rename('energy_forecast')
        .rename_axis(['generator', 'interval']).reset_index().drop('interval', axis=1).set_index('generator'))

    df_e_compare = pd.concat([df_e_actual, df_e_forecast], axis=1, sort=True)

    fig, ax = plt.subplots()
    df_e_compare.plot.scatter(x='energy_observed', y='energy_forecast', ax=ax)
    ax.plot([0, 120000], [0, 120000])
    # for k, v in df_e_compare.iterrows():
    #     ax.annotate(k, v)
    plt.show()
