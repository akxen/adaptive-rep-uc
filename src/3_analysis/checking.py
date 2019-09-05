"""Check preliminary results"""

import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'base'))

import pandas as pd
import matplotlib.pyplot as plt

from data import ModelData


class CheckResults:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.data = ModelData()

    def load_interval_results(self, year, week, day):
        """Load interval results"""

        with open(os.path.join(self.output_dir, f'interval_{year}_{week}_{day}.pickle'), 'rb') as f:
            results = pickle.load(f)

        return results

    def load_all_results(self):
        """Load all interval results"""

        # Container for all results
        all_results = {}

        # Get file names
        files = [f for f in os.listdir(self.output_dir) if 'pickle' in f]

        for f in files:
            # Load results
            with open(os.path.join(self.output_dir, f), 'rb') as g:
                results = pickle.load(g)

            # Place in dictionary
            all_results = {**all_results, **results}

        return all_results

    def get_generator_results(self, year, month, day, key):
        """Get results indexed by generator and interval (g, t)"""

        # Load results
        results = self.load_interval_results(year, month, day)

        # Get generator results
        df = (pd.Series(results[list(results.keys())[0]][key]).rename(key)
              .rename_axis(['generator', 'interval']).reset_index()
              .pivot(index='interval', columns='generator', values=key))

        return df


if __name__ == '__main__':
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'output')

    check = CheckResults(output_directory)

    year, week, day = 2017, 1, 1

    r = check.load_interval_results(year, week, day)
    df_u = check.get_generator_results(year, week, day, 'u')
    df_v = check.get_generator_results(year, week, day, 'v')
    df_w = check.get_generator_results(year, week, day, 'w')
    df_p_total = check.get_generator_results(year, week, day, 'p_total')
    df_p_in = check.get_generator_results(year, week, day, 'p_in')
    df_p_out = check.get_generator_results(year, week, day, 'p_out')
