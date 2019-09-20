"""Generator energy forecasting"""

import os

import pandas as pd

from data import ModelData
from analysis import AnalyseResults


class Forecast:
    def __init__(self):
        self.data = ModelData()
        self.analysis = AnalyseResults()

    def get_energy_forecast_persistence(self, output_dir, year, week, n_intervals, eligible_generators):
        """Get persistence based energy forecast. Energy output in previous week assumed same for following weeks."""

        # Container for energy output DataFrames
        dfs = []

        for day in range(1, 8):
            df_o = self.analysis.get_generator_interval_results(output_dir, 'e', year, week-1, day)
            dfs.append(df_o)

        # Concatenate DataFrames
        df_c = pd.concat(dfs)

        # Energy forecast
        energy_forecast = {(g, c, 1): v for g, v in df_c.sum().to_dict().items() for c in range(1, n_intervals + 1)
                           if g in eligible_generators}

        # Assume probability = 1 for each scenario (only one scenario per calibration interval for persistence forecast)
        probabilities = {(i, 1): float(1) for i in range(1, n_intervals + 1)}

        return energy_forecast, probabilities


if __name__ == '__main__':
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output')
    y, w, intervals = 2018, 2, 6

    forecast = Forecast()
    e_forecast = forecast.get_energy_forecast_persistence(output_directory, y, w, intervals, ['AGLHAL', 'TORRA1'])


