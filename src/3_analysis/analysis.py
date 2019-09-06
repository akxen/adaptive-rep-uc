"""Analyse results"""

import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'base'))

import pandas as pd
import matplotlib.pyplot as plt

from data import ModelData


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


if __name__ == '__main__':
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'output')

    model_baselines = get_baselines(output_directory)

    df_b = pd.Series(model_baselines)