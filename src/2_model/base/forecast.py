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
        energy_forecast = {(g, 1, c): v for g, v in df_c.sum().to_dict().items() for c in range(1, n_intervals + 1)
                           if g in eligible_generators}

        # Assume probability = 1 for each scenario (only one scenario per calibration interval for persistence forecast)
        probabilities = {(g, 1): float(1) for g in df_c.sum().to_dict().keys() if g in eligible_generators}

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
            # for duid in [duid]:
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
                        last_observation = lagged.loc[:, [f'{duid}_lag_{i}' for i in range(0, lags + 1)]].iloc[
                            -1].values
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

    def get_weekly_energy(self, year, week, start_year=2018):
        """Compute weekly generator energy output for all weeks preceding 'year' and 'week'"""

        df = self.get_observed_energy(output_directory, range(start_year, year + 1), week)
        energy = df.groupby(['year', 'week']).sum()

        return energy

    def get_max_energy(self, duid):
        """Compute max weekly energy output if generator output at max capacity for whole week"""

        # Max weekly energy output
        if duid in self.data.generators.index:
            max_energy = self.data.generators.loc[duid, 'REG_CAP'] * 24 * 7

        # Must spend at least half the time charging if a storage unit (assumes charging and discharging rates are same)
        elif duid in self.data.storage.index:
            max_energy = (self.data.storage.loc[duid, 'REG_CAP'] * 24 * 7) / 2

        else:
            raise Exception(f'Unidentified DUID: {duid}')

        return max_energy

    def get_scenarios(self, energy, duid, n_intervals, n_clusters, n_random_paths):
        """
        Get randomised scenarios

        Params
        ------
        energy : pandas DataFrame
            Weekly energy output for each generator

        duid : str
            Dispatchable Unit Identifier for generator

        n_intervals : int
            Number of forecast intervals

        n_clusters : int
            Number of centroid clusters

        n_random_paths : int
            Number of randomised scenario paths. These will be clustered to yield the final set of scenarios.

        Returns
        -------
        scenario_energy : dict
            Energy output paths over future calibration intervals

        scenario_weights : dict
            Probability corresponding to each scenario
        """

        # Weekly energy output (series) for a given DUID
        s = energy[duid]

        # Max possible energy output over coming week
        max_energy = self.get_max_energy(duid)

        # Compute log pct change
        log_pct_change = np.log(1 + s.pct_change())

        # Compute mean, variance, drift term, and standard deviation
        mean = log_pct_change.mean()
        var = log_pct_change.var()
        drift = mean - (0.5 * var)
        std = log_pct_change.std()

        # Randomised paths
        random_paths = np.exp(drift + std * norm.ppf(np.random.rand(n_intervals, n_random_paths)))

        # NaNs occur when dividing 0/0 (e.g. generator is off over consecutive weeks
        random_paths[np.isnan(random_paths)] = 0

        # Last observed values for weekly energy output
        last_observation = s.iloc[-1]

        # Initialise array in which randomised paths will be stored (all zeros)
        energy_paths = np.zeros((n_intervals + 1, n_random_paths))

        # First row in array will be last observed value
        energy_paths[0] = energy_paths[0] + last_observation

        for i in range(1, n_intervals + 1):
            # Compute energy over forecast horizon
            energy_paths[i] = energy_paths[i - 1] * random_paths[i - 1]

            # Ensure energy limit observed
            energy_paths[i][energy_paths[i] > max_energy] = max_energy

        # Construct K-means classifier and fit to randomised energy paths
        k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(energy_paths.T)

        # Get cluster centroids (these are will be the energy paths used in the analysis
        clusters = k_means.cluster_centers_

        # Get scenario energy in format to be consumed by model
        scenario_energy = {(duid, s, c): clusters[s - 1][c] for s in range(1, n_clusters + 1)
                           for c in range(1, n_intervals + 1)}

        # Determine number of randomised paths assigned to each cluster
        cluster_assignment = np.unique(k_means.labels_, return_counts=True)

        # Compute probabilities for each scenario
        probabilities = {cluster + 1: cluster_assignment[1][index] / n_random_paths
                         for index, cluster in enumerate(cluster_assignment[0])}

        # May need to pad scenario probabilities (e.g. if only one cluster returned)
        for s in set(range(1, n_clusters + 1)).difference(set([i + 1 for i in cluster_assignment[0]])):
            # If cluster ID missing, set probability to zero
            probabilities[s] = 0

        # Weighting assigned to each scenario (random paths assigned to scenario / total number of randomised paths)
        scenario_weights = {(duid, s): probabilities[s] for s in range(1, n_clusters + 1)}

        return scenario_energy, scenario_weights

    def get_scenarios_historic_update(self, energy, duid, n_intervals, n_random_paths, n_clusters):
        """Randomly sample based on difference in energy output between successive weeks"""

        # Max energy output
        max_energy = self.get_max_energy(duid)

        # Last observation for given DUID
        last_observation = energy[duid].iloc[-1]

        # Container for all random paths
        energy_paths = []

        for r in range(1, n_random_paths + 1):
            # Container for randomised calibration interval observations
            interval = [last_observation]
            for c in range(1, n_intervals + 1):
                # Update
                update = np.random.normal(energy[duid].diff(1).mean(), energy[duid].diff(1).std())

                # New observation
                new_observation = last_observation + update

                # Check that new observation doesn't violate upper and lower revenue bounds
                if new_observation > max_energy:
                    new_observation = max_energy
                elif new_observation < 0:
                    new_observation = 0

                # Append to container
                interval.append(new_observation)

            # Append to random paths container
            energy_paths.append(interval)

        # Construct K-means classifier and fit to randomised energy paths
        k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(energy_paths)

        # Get cluster centroids (these are will be the energy paths used in the analysis
        clusters = k_means.cluster_centers_

        # Get scenario energy in format to be consumed by model
        scenario_energy = {(duid, s, c): clusters[s - 1][c] for s in range(1, n_clusters + 1)
                           for c in range(1, n_intervals + 1)}

        # Determine number of randomised paths assigned to each cluster
        assignment = np.unique(k_means.labels_, return_counts=True)

        # Weighting dependent on number of paths assigned to each scenarios
        scenario_weights = {k + 1: v / n_random_paths for k, v in zip(assignment[0], assignment[1])}

        return scenario_energy, scenario_weights


if __name__ == '__main__':
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output', '3_calibration_intervals')

    # persistence_forecast = PersistenceForecast()
    forecast = MonteCarloForecast()

    # Weekly energy output for a given week
    energy = forecast.get_weekly_energy(2018, 30, start_year=2018)

    duid = 'GANNBG1'
    s_energy, s_weights = forecast.get_scenarios_historic_update(energy, duid, n_intervals=3, n_random_paths=500,
                                                                 n_clusters=5)

    n_clusters, n_intervals = 5, 3
    scens = [[s_energy[(duid, s, c)] for c in range(1, n_intervals + 1)] for s in range(1, n_clusters + 1)]

    x_obs = [x for x in range(energy.shape[0])]
    x_pred = [x for x in range(x_obs[-1] + 1, x_obs[-1] + n_intervals + 1)]

    fig, ax = plt.subplots()
    ax.plot(x_obs, energy[duid].values)
    for s in scens:
        ax.plot(x_pred, s, color='r')
    plt.show()
