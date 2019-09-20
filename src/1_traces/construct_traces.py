"""Construct model input traces"""

import os
import pickle

import MySQLdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def load_data(table, use_cache=False, cache_dir=os.path.join(os.path.dirname(__file__), 'tmp')):
    """Load data from MySQL database or from cache directory"""

    if use_cache:
        # Read cached file
        df = pd.read_pickle(os.path.join(cache_dir, f'{table}.pickle'))

    else:
        # Establish connection to database
        conn = MySQLdb.connect(host=os.environ['host'], user=os.environ['user'], db=os.environ['database'],
                               passwd=os.environ['password'])

        # Query to execute
        sql = f"""
        SELECT * FROM archive.{table} WHERE SETTLEMENTDATE >= '2017-01-01 00:00:05' AND 
        SETTLEMENTDATE <= '2019-01-01 00:00:00' ORDER BY SETTLEMENTDATE DESC;
        """

        # Read sql data into pandas DataFrame
        df = pd.read_sql(sql, con=conn)

        # Save cached file
        with open(os.path.join(cache_dir, f'{table}.pickle'), 'wb') as f:
            pickle.dump(df, f)

        # Close database connection
        conn.close()

    return df


def get_dispatch_data(overlap, use_cache=True):
    """Convert DataFrame to hourly resolution"""

    # Load dispatch unit SCADA data for 2017 and 2018
    df = load_data('dispatch_unit_scada', use_cache=use_cache)

    # Pivot
    df_p = df.pivot(index='SETTLEMENTDATE', columns='DUID', values='SCADAVALUE')

    # Re-sampled to hourly resolution
    df_r = df_p.resample('1h', label='right', closed='right').mean()

    # Add year, week, day
    calendar = df_r.apply(lambda x: pd.Series(x.name.isocalendar(), index=['year', 'week', 'day']), axis=1)

    # For each day find the first timestamp. Note: +1 because want to include t_0 (used for initialisation).
    intervals = (calendar.groupby(['year', 'week', 'day'])
                 .apply(lambda x: pd.date_range(start=x.index.min(), periods=24 + overlap + 1, freq='1h')))

    # Dispatch information for each generator
    dispatch = (intervals.map(lambda x: df_r.reindex(x).reset_index().drop('index', axis=1).T.stack()
                              .to_dict()).to_dict())

    return dispatch


def get_demand_data(overlap, use_cache=True):
    """Get demand data for each NEM zone"""

    # Load demand data for 2017 and 2018
    df = load_data('dispatchregionsum', use_cache=use_cache)

    # Pivot
    df_p = (df.drop_duplicates(subset=['SETTLEMENTDATE', 'REGIONID'], keep='last')
            .pivot(index='SETTLEMENTDATE', columns='REGIONID', values='TOTALDEMAND'))

    # Re-sampled to hourly resolution. Rename column axis to 'NEM_REGION'. Same as in network nodes dataset
    df_r = df_p.resample('1h', label='right', closed='right').mean().rename_axis('NEM_REGION', axis=1)

    grid_data_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'data', 'files',
                                 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603')

    df_n = pd.read_csv(os.path.join(grid_data_dir, 'network', 'network_nodes.csv'))

    demand_proportion = df_n.groupby(['NEM_REGION', 'NEM_ZONE'])['PROP_REG_D'].sum()

    df_d = df_r.apply(lambda x: x * demand_proportion, axis=1).droplevel(0, axis=1)

    # Add year, week, day
    calendar = df_d.apply(lambda x: pd.Series(x.name.isocalendar(), index=['year', 'week', 'day']), axis=1)

    # For each day find the first timestamp
    intervals = (calendar.groupby(['year', 'week', 'day'])
                 .apply(lambda x: pd.date_range(start=x.index.min(), periods=24 + overlap + 1, freq='1h')))

    # Dispatch information for each generator
    demand = (intervals.map(lambda x: df_d.reindex(x).reset_index().drop('index', axis=1).T.stack()
                            .to_dict()).to_dict())

    return demand


if __name__ == '__main__':
    # Interval overlap
    interval_overlap = 17

    # Dispatch data
    dispatch_data = get_dispatch_data(interval_overlap, use_cache=True)
    with open(os.path.join(os.path.dirname(__file__), 'output', f'dispatch_{24+interval_overlap+1}.pickle'), 'wb') as f:
        pickle.dump(dispatch_data, f)

    # Demand data
    demand_data = get_demand_data(interval_overlap, use_cache=True)
    with open(os.path.join(os.path.dirname(__file__), 'output', f'demand_{24+interval_overlap+1}.pickle'), 'wb') as f:
        pickle.dump(demand_data, f)
