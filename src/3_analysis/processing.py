"""Process model results"""

import os
import sys
import pickle

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'base'))

from data import ModelData
from uc import UnitCommitment


class Results:
    def __init__(self, root_output_dir=os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'output')):
        self.root_output_dir = root_output_dir

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


if __name__ == '__main__':
    data = ModelData()
    process = Results()
    uc = UnitCommitment()
    m = uc.construct_model(overlap=17)
    # m_r = process.load_mpc_results('revenue_target', 2018, 20)
    # c_rev = process.get_cumulative_scheme_revenue('revenue_target')

    i_r = process.load_interval_results('revenue_target', 2018, 2, 2)

    prices = {}
    overlap = 17

    for year in [2018]:
        for week in range(1, 20):
            for day in range(1, 8):
                interval_results = process.load_interval_results('revenue_target', year, week, day)

                for k, v in interval_results['POWER_BALANCE'].items():
                    if (week == 1) and (day == 1) and (k[1] <= 24):
                        prices[(year, week, day, k[1], k[0])] = v

                    elif (week >= 1) and (k[1] > overlap) and (k[1] <= 24):
                        prices[(year, week, day, k[1], k[0])] = v

                    elif (week >= 1) and (k[1] > 24):
                        if day == 7:
                            prices[(year, week + 1, 1, k[1] - 24, k[0])] = v
                        else:
                            prices[(year, week, day + 1, k[1] - 24, k[0])] = v

    # df_p = pd.Series(prices).rename_axis(['year', 'week', 'day', 'hour', 'zone']).rename('price').drop((2018, 53))
    df_p = pd.Series(prices).rename_axis(['year', 'week', 'day', 'hour', 'zone']).rename('price')
    df_p = df_p.reset_index().pivot_table(index=['year', 'week', 'day', 'hour'], columns='zone', values='price')

    i_r1 = process.load_interval_results('bau', 2018, 2, 1)
    i_r2 = process.load_interval_results('bau', 2018, 2, 2)

    p_total = (pd.Series(i_r2[(2018, 2, 2)]['p_total']).rename_axis(['DUID', 'hour']).to_frame('p_total')
               .join(data.generators[['NEM_ZONE']], how='left'))

    r_up = (pd.Series(i_r2[(2018, 2, 2)]['r_up']).rename_axis(['DUID', 'hour']).to_frame('r_up')
            .join(data.generators[['NEM_ZONE']], how='left'))
