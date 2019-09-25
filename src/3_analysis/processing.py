"""Process model results"""

import os
import sys
import pickle

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'base'))

from data import ModelData


class Results:
    def __init__(self, root_output_dir=os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'output')):
        self.root_output_dir = root_output_dir
        self.data = ModelData()

    def load_interval_results(self, case, year, week, day):
        """Load results for a given interval"""

        with open(os.path.join(self.root_output_dir, case, f'interval_{year}_{week}_{day}.pickle'), 'rb') as f:
            results = pickle.load(f)

        return results

    def load_mpc_results(self, case, year, week):
        """Load MPC program results for a given interval"""

        with open(os.path.join(self.root_output_dir, case, f'mpc_{year}_{week}.pickle'), 'rb') as f:
            results = pickle.load(f)

        return results

    def get_cumulative_scheme_revenue(self, case):
        """Get cumulative scheme revenue for a given case"""

        # Initialise container for cumulative scheme revenue
        cumulative_scheme_revenue = {}

        for year in [2018]:
            for week in range(2, 53):
                mpc_results = self.load_mpc_results(case, year, week)

                # Get cumulative scheme revenue
                cumulative_scheme_revenue[(year, week)] = mpc_results['parameters']['revenue_start']

        return cumulative_scheme_revenue

    def get_case_parameters(self, case):
        """Get case parameters"""

        with open(os.path.join(self.root_output_dir, case, 'parameters.pickle'), 'rb') as f:
            parameters = pickle.load(f)

        return parameters

    def get_interval_prices(self, case):
        """Get interval prices"""

        # Load case parameters
        parameters = self.get_case_parameters(case)

        # Container for interval prices
        interval_prices = {}

        for year in parameters['years']:
            for week in parameters['weeks']:
                for day in parameters['days']:
                    # Load interval results
                    interval_results = self.load_interval_results(case, year, week, day)

                    # Extract prices
                    for k, v in interval_results['POWER_BALANCE'].items():
                        if (week == 1) and (day == 1) and (k[1] <= 24):
                            interval_prices[(year, week, day, k[1], k[0])] = v

                        elif (week >= 1) and (k[1] > parameters['overlap_intervals']) and (k[1] <= 24):
                            interval_prices[(year, week, day, k[1], k[0])] = v

                        elif (week >= 1) and (k[1] > 24):
                            if day == 7:
                                interval_prices[(year, week + 1, 1, k[1] - 24, k[0])] = v
                            else:
                                interval_prices[(year, week, day + 1, k[1] - 24, k[0])] = v

        # Convert to series
        df = pd.Series(interval_prices).rename_axis(['year', 'week', 'day', 'hour', 'zone']).rename('price')

        # Pivot and construct DataFrame
        df = df.reset_index().pivot_table(index=['year', 'week', 'day', 'hour'], columns='zone', values='price')

        return df

    def get_interval_demand(self):
        """Get demand for each dispatch interval"""

        # Container for re-indexed demand data
        new_demand = {}

        # Construct series such that dict keys and values are in format: (year, week, day, hour, zone): demand_value
        for (year, week, day), values in self.data.demand.items():
            for (zone, hour), demand in values.items():
                new_demand[(year, week, day, hour, zone)] = demand

        # Construct pandas series
        df = pd.Series(new_demand).rename_axis(['year', 'week', 'day', 'hour', 'zone']).rename('demand')

        # Construct DataFrame and apply pivot
        df = df.reset_index().pivot_table(index=['year', 'week', 'day', 'hour'], columns='zone', values='demand')

        return df

    def get_week_prices(self, case):
        """Get zone and NEM average price for each week"""

        # Get prices for each week for a given case. Drop last (incomplete) week.
        interval_prices = self.get_interval_prices(case)
        interval_prices = interval_prices.drop((2018, 53))

        # Get total demand for each NEM zone in each dispatch interval
        interval_demand = self.get_interval_demand()
        interval_demand = interval_demand.reindex(interval_prices.index)

        # Compute revenue from electricity sales
        revenue = interval_demand.mul(interval_prices)

        # Weekly average price and total energy output
        week_revenue = revenue.groupby(['year', 'week']).sum()
        week_energy = interval_demand.groupby(['year', 'week']).sum()

        # Average price in each zone, and for NEM as a whole for each week
        zone_price = week_revenue.div(week_energy).dropna(how='all')
        nem_price = week_revenue.sum(axis=1).div(week_energy.sum(axis=1))

        return nem_price, zone_price

    def get_baselines_and_revenue(self, case):
        """Get model baselines"""

        # MPC result files
        files = [f for f in os.listdir(os.path.join(self.root_output_dir, case)) if 'mpc' in f]

        # Containers for model baselines and scheme revenue
        baselines = {}
        revenue = {}

        min_year = 999999
        min_week = 999999
        start_baseline = 0

        for f in files:
            # Get year and week from filename and load MPC results
            year, week = int(f.split('_')[1]), int(f.split('_')[2].replace('.pickle', ''))
            results = self.load_mpc_results(case, year, week)

            # Update min year and min week
            if year <= min_year:
                min_year = year
                if week < min_week:
                    min_week = week
                    start_baseline = results['parameters']['baseline_start']

            # Add baselines and cumulative scheme revenue to containers
            baselines[(year, week)] = results['baseline_trajectory'][1]
            revenue[(year, week)] = results['parameters']['revenue_start']

        # Add starting baseline and starting scheme revenue
        baselines[(min_year, min_week - 1)] = start_baseline
        revenue[(min_year, min_week - 1)] = 0

        # Convert to pandas series and sort index
        df_b = pd.Series(baselines).rename_axis(['year', 'week']).rename('baseline').sort_index()
        df_r = pd.Series(revenue).rename_axis(['year', 'week']).rename('cumulative_revenue').sort_index()

        return df_b, df_r

    def get_year_emissions(self, case):
        """Get total emissions in a given year"""

        files = [f for f in os.listdir(os.path.join(self.root_output_dir, case)) if 'interval' in f]

        # Total emissions
        day_emissions = {}

        for f in files:
            # Get year, week, and day from filename
            year, week, day = int(f.split('_')[1]), int(f.split('_')[2]), int(f.split('_')[3].replace('.pickle', ''))

            # Interval results
            interval_results = self.load_interval_results(case, year, week, day)

            # Total emissions
            day_emissions[(year, week, day)] = interval_results['DAY_EMISSIONS']

        # Convert to pandas series
        df = pd.Series(day_emissions).rename_axis(['year', 'week', 'day']).rename('day_emissions').sort_index()

        return df


if __name__ == '__main__':
    # Object used to process model data
    process = Results()

    case_name = 'bau'

    params = process.get_case_parameters(case_name)
    # week_nem_price, week_zone_price = process.get_week_prices(case_name)
    r = process.load_mpc_results('revenue_target', 2018, 2)
    i = process.load_interval_results(case_name, 2018, 2, 1)

    bau_emissions = process.get_year_emissions('bau')
    tax_emissions = process.get_year_emissions('carbon_tax')

    bas, rev = process.get_baselines_and_revenue('emissions_intensity_shock')

    fig, ax = plt.subplots()
    bau_emissions.cumsum().plot(ax=ax, color='blue')
    tax_emissions.cumsum().plot(ax=ax, color='red')
    plt.show()

    # bas, rev = process.get_baselines_and_revenue(case_name)
    # bas.plot()
    # plt.show()
    # rev.plot()
    # plt.show()
