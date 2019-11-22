"""Generator energy forecasting"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from data import ModelData
from analysis import AnalyseResults

np.random.seed(10)


class PersistenceForecast:
    def __init__(self):
        self.data = ModelData()
        self.analysis = AnalyseResults()

    def get_energy_forecast_persistence(self, output_dir, year, week, n_intervals, eligible_generators):
        """
        Get persistence based energy forecast. Energy output in previous week assumed same for following weeks.

        Params
        ------
        output_dir : str
            Path to directory containing output files

        year : int


        """

        # Take into account end-of-year transition
        if week == 1:
            previous_interval_year = year - 1
            previous_interval_week = 52
        else:
            previous_interval_year = year
            previous_interval_week = week - 1

        # Container for energy output DataFrames
        dfs = []

        for day in range(1, 8):
            df_o = self.analysis.get_generator_interval_results(output_dir, 'e', previous_interval_year,
                                                                previous_interval_week, day)
            dfs.append(df_o)

        # Concatenate DataFrames
        df_c = pd.concat(dfs)

        # Energy forecast
        energy_forecast = {(g, c, 1): v for g, v in df_c.sum().to_dict().items() for c in range(1, n_intervals + 1)
                           if g in eligible_generators}

        # Assume probability = 1 for each scenario (only one scenario per calibration interval for persistence forecast)
        # probabilities = {(g, c, 1): float(1) for c in range(1, n_intervals + 1)}
        probabilities = {(g, c, 1): float(1) for g in df_c.sum().to_dict().keys() for c in range(1, n_intervals + 1)
                         if g in eligible_generators}

        return energy_forecast, probabilities


class ProbabilisticForecast:
    def __init__(self):
        self.data = ModelData()
        self.analysis = AnalyseResults()

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

    def construct_dataset(self, years, week, lags=6, future_intervals=3):
        """Construct dataset to be used in quantile regression models for a given DUID"""

        # Observed generator energy output for each dispatch interval
        observed = self.get_observed_energy(output_directory, years, week)

        # Aggregate energy output by week
        weekly_energy = observed.groupby(['year', 'week']).sum()

        # Lagged values
        lagged = pd.concat([weekly_energy.shift(i).add_suffix(f'_lag_{i}') for i in range(0, lags + 1)], axis=1)

        # Future values
        future = pd.concat([weekly_energy.shift(-i).add_suffix(f'_future_{i}') for i in range(1, future_intervals + 1)],
                           axis=1)

        # Re-index so lagged and future observations have the same index
        # new_index = lagged.index.intersection(future.index).sort_values()
        # lagged = lagged.reindex(new_index)
        # future = future.reindex(new_index)

        return lagged, future

    def fit_model(self, x, y, duid):
        pass

    def construct_quantile_regression_models(self, years, week, lags=6, future_intervals=3):
        """Construct regression models for each"""

        # Construct dataset
        lagged, future = self.construct_dataset(years, week)

        # DUIDs
        duids = list(set([i.split('_future')[0] for i in future.columns]))
        duids.sort()

        # Container for quantile regression results
        results = {}

        # Run model for each quantile
        for duid in duids:
        # for duid in ['ARWF1']:
            results[duid] = {}

            # Lagged values
            x = pd.concat([lagged.loc[:, f'{duid}_lag_{i}'] for i in range(0, lags + 1)], axis=1)
            x = x.dropna()

            # For each future interval range
            for f in range(1, future_intervals + 1):
                results[duid][f] = {}

                # Split independent and dependent variables
                y = future[f'{duid}_future_{f}']
                y = y.dropna()

                # Ensure index is the same
                new_index = y.index.intersection(x.index).sort_values()
                x = x.reindex(new_index)
                y = y.reindex(new_index)

                # Run model for each quantile
                for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
                    # print(f'Fitting model: duid={duid}, future_interval={f}, quantile={q}')

                    try:
                        # Construct and fit model
                        m = sm.QuantReg(y, x)
                        res = m.fit(q=q)

                        # Make prediction for last time point
                        last_observation = lagged.loc[:, [f'{duid}_lag_{i}' for i in range(0, lags + 1)]].iloc[-1].values
                        pred = res.predict(last_observation)[0]
                        results[duid][f][q] = pred

                    except ValueError:
                        results[duid][f][q] = None
                        # print(f'Failed for: duid={duid}, quantile={q}')

        return results


class MonteCarloForecast:
    def __init__(self):
        self.data = ModelData()
        self.analysis = AnalyseResults()

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


if __name__ == '__main__':
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output', '3_calibration_intervals')
    y, w, intervals = 2018, 2, 3

    persistence_forecast = PersistenceForecast()
    # persistence_energy = persistence_forecast.get_energy_forecast_persistence(output_directory, )

    forecast = MonteCarloForecast()
    # e_forecast = forecast.get_energy_forecast_persistence(output_directory, y, w, intervals, ['AGLHAL', 'TORRA1'])
    # lagged, future = forecast.construct_dataset([2018], 30, lags=2)
    # models = forecast.construct_quantile_regression_models([2018], 30, lags=2, future_intervals=3)

    df = forecast.get_observed_energy(output_directory, [2018], 30)
    df_wk = df.groupby(['year', 'week']).sum()

    max_energy = forecast.data.generators.loc['ARWF1', 'REG_CAP'] * 24 * 7

    # change = df_wk.pct_change().fillna(0) + 1
    energy = df_wk['ARWF1'].copy()
    log_pct_change = np.log(1 + energy.pct_change())
    mean = log_pct_change.mean()
    var = log_pct_change.var()
    drift = mean - (0.5 * var)
    stdev = log_pct_change.std()

    forecast_intervals = 3
    paths = 500

    energy_paths = np.exp(drift + stdev * norm.ppf(np.random.rand(forecast_intervals, paths)))

    S0 = energy.iloc[-1]
    path_list = np.zeros((forecast_intervals + 1, paths))
    path_list[0] = path_list[0] + S0

    for i in range(1, forecast_intervals + 1):
        path_list[i] = path_list[i-1] * energy_paths[i-1]

        # Ensure energy limit observed
        path_list[i][path_list[i] > max_energy] = max_energy

    x_obs = range(1, 11)
    y_obs = energy.iloc[-10:].values

    x_scen = range(10, 14)
    y_scen = [path_list[:, i] for i in range(0, paths)]

    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(path_list.T)
    clusters = kmeans.cluster_centers_

    fig, ax = plt.subplots()
    ax.plot(x_obs, y_obs, color='b')

    for i in range(0, paths):
        ax.plot(x_scen, y_scen[i], color='r', alpha=0.5)

    for i in range(0, 10):
        ax.plot(x_scen, clusters[i], color='k')
    plt.show()

    # (g, c, s)
    scenario_energy = {('ARWF1', c, s): clusters[s-1][c] for c in range(1, forecast_intervals + 1) for s in range(1, n_clusters + 1)}

    cluster_assignment = np.unique(kmeans.labels_, return_counts=True)

    scenario_probability = {('ARWF1', s): cluster_assignment[1][s-1] / paths for s in range(1, n_clusters + 1)}
