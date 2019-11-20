"""Generator energy forecasting"""

import os

import pandas as pd
import statsmodels.formula.api as sm

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

    def get_probabilistic_energy_forecast(self, output_dir, year, week, n_intervals, eligible_generators, n_clusters):
        """Construct probabilistic forecast for energy output in future weeks for each generator"""

        # Get generator results
        energy = self.get_observed_energy(output_dir, range(2017, year + 1), week)

        # Construct regression models

        # Generate scenarios from regression model

        # Cluster scenarios

        # Get energy forecasts for each scenario

        # Get probabilities for each scenario

        pass

    def get_observed_energy(self, output_dir, years, week):
        """Get observed energy for all years and weeks up until the defined week and year"""

        # Container for energy output DataFrames
        dfs = []

        for y in years:
            for w in range(1, week + 1):
                for d in range(1, 8):
                    df_o = self.analysis.get_generator_interval_results(output_dir, 'e', y, w, d)
                    dfs.append(df_o)

        # Concatenate DataFrames
        df_c = pd.concat(dfs)

        return df_c

    def construct_regression_models(self, energy):
        """Construct regression models for each"""
        pass


if __name__ == '__main__':
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output', '3_calibration_intervals')
    y, w, intervals = 2018, 2, 3

    forecast = Forecast()
    # e_forecast = forecast.get_energy_forecast_persistence(output_directory, y, w, intervals, ['AGLHAL', 'TORRA1'])
    obs = forecast.get_observed_energy(output_directory, [2018], 2)

    weekly_energy = obs.groupby(['year', 'week']).sum()

    generator_energy = weekly_energy.loc[:, 'YWPS1']

    model_data = pd.concat([generator_energy.to_frame('observed')] + [generator_energy.shift(lag).to_frame(f'lag_{lag}') for lag in range(1, 3)], axis=1)

    model_result = sm.ols(formula='observed ~ lag_1 + lag_2', data=model_data).fit()

