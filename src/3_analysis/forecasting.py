"""Demand probabilistic forecasting"""

import os
import pickle

import MySQLdb
import MySQLdb.cursors
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import statsmodels.api as sm
import matplotlib.pyplot as plt

load_dotenv()


def get_data(sql, filename, output_dir, use_cache):
    """Load data"""

    if use_cache:
        with open(os.path.join(output_dir, filename), 'rb') as f:
            data = pickle.load(f)

    else:
        # Database parameters
        host = os.environ['LOCAL_HOST']
        schema = os.environ['LOCAL_SCHEMA']
        user = os.environ['LOCAL_USER']
        password = os.environ['LOCAL_PASSWORD']

        # MySQL database connection and SQL query to get demand data
        conn = MySQLdb.connect(host=host, db=schema, user=user, passwd=password, cursorclass=MySQLdb.cursors.DictCursor)
        cur = conn.cursor()

        # Execute query and return data
        cur.execute(sql)
        res = cur.fetchall()

        # Convert to DataFrame
        data = pd.DataFrame(res)

        # Save demand data
        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(data, f)

    return data


def reindex_data(df):
    """Re-index pivoted DataFrame to different time resolutions"""

    # Regional energy demand at hourly resolution (5 min intervals so MWh = MW x (1/12))
    df_h = df.mul(1/12).resample('1h', label='right', closed='right').sum()

    # Regional energy demand at daily resolution
    df_d = df_h.resample('1d', closed='right').sum()

    # Get ISO calendar dates and add them to the DataFrame
    iso_dates = df_d.apply(lambda x: pd.Series({k: v for k, v in zip(('year', 'week', 'day'), x.name.isocalendar())}),
                           axis=1)
    df_dd = pd.concat([df_d, iso_dates], axis=1)

    # Weekly energy demand (first and last weeks may not be comprised of a total of 7 days - may need to remove).
    df_w = df_dd.groupby(['year', 'week']).sum().drop('day', axis=1)

    return df, df_h, df_dd, df_w


def process_demand_data(output_dir, use_cache):
    """Process demand data"""

    # Get data
    sql = "SELECT SETTLEMENTDATE, REGIONID, TOTALDEMAND FROM dispatchregionsum"
    df = get_data(sql, 'region_demand.pickle', output_dir, use_cache)

    # Pivot so regions are columns, and index is timestamp
    df_p = (df.drop_duplicates(subset=['SETTLEMENTDATE', 'REGIONID'], keep='last')
            .pivot(index='SETTLEMENTDATE', columns='REGIONID', values='TOTALDEMAND').astype(float))

    # Data at different time resolutions
    df_p, df_h, df_dd, df_w = reindex_data(df_p)

    return df, df_h, df_dd, df_w


def process_dispatch_data(output_dir, use_cache):
    """Process dispatch data for 2018"""

    if not use_cache:
        # MySQL command to be executed
        command = f"""mysql -h {os.environ['LOCAL_HOST']} -u {os.environ['LOCAL_USER']} -D {os.environ['LOCAL_SCHEMA']} -p{os.environ['LOCAL_PASSWORD']} --batch --quick -e "SELECT * FROM dispatch_unit_scada WHERE SETTLEMENTDATE >= '2018-01-01 00:05:00' AND SETTLEMENTDATE <= '2019-01-01 00:00:00'" > ./output/dispatch_data.csv"""

        # Download chunk
        os.system(command)

    # Load dispatch data
    df = pd.read_csv(os.path.join(output_dir, 'dispatch_data.csv'), delimiter='\t')

    # Convert column to datetime
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

    # Apply pivot and convert all data to type float
    df_p = df.pivot(index='SETTLEMENTDATE', columns='DUID', values='SCADAVALUE').astype(float)

    # Data at different time resolutions
    df_p, df_h, df_dd, df_w = reindex_data(df_p)

    return df_p, df_h, df_dd, df_w


def construct_dataset(df, past_lags, future_intervals):
    """Construct demand dataset"""

    # Total NEM demand for a given week
    df_t = df.sum(axis=1).rename('total_demand')

    # Add an interval ID
    df_t = df_t.reset_index().rename_axis('interval').set_index(['year', 'week'], append=True)

    # Get lags
    for i in range(1, past_lags + 1):
        df_t[f'lag_{i}'] = df_t['total_demand'].shift(i)

    # Get lags
    for i in range(1, future_intervals + 1):
        df_t[f'future_{i}'] = df_t['total_demand'].shift(-i)

    # Rename total_demand to lag_0
    total = df_t.rename(columns={'total_demand': 'lag_0'})

    # Drop missing values
    total = total.dropna()

    return total


def train_test_split(df, train_proportion=0.7):
    """Split dataset into training and testing sets"""

    # Index before which data will be used to training, and after which will be used for testing
    split_index = int(df.shape[0] * train_proportion)

    training = df.loc[:split_index, :]
    testing = df.loc[split_index+1:, :]

    x_training = training.loc[:, training.columns.str.contains('lag')]
    y_training = training.loc[:, training.columns.str.contains('future')]

    x_testing = training.loc[:, testing.columns.str.contains('lag')]
    y_testing = training.loc[:, testing.columns.str.contains('future')]

    return x_training, y_training, x_testing, y_testing


def get_weekly_dispatch_data(output_dir, use_cache):
    """Get weekly dispatch data for 2018"""

    if use_cache:
        with open(os.path.join(output_dir, 'dispatch_weekly.pickle'), 'rb') as f:
            return pickle.load(f)

    else:
        # Process dispatch data
        dfs = []
        for month in [f'{i.year}-{i.month:02}' for i in pd.date_range('2018-01', periods=12, freq='M')]:
            print(month)
            _, _, _, weekly = process_dispatch_data(month, output_dir, use_cache=use_cache)
            dfs.append(weekly)

        # Concatenate weekly aggregate dispatch data for each generator
        df_c = pd.concat(dfs, sort=True)

        with open(os.path.join(output_dir, 'dispatch_weekly.pickle'), 'wb') as f:
            pickle.dump(df_c, f)

        return df_c


if __name__ == '__main__':
    # Output directory
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Dispatch data
    _, _, _, dispatch = process_dispatch_data(output_directory, use_cache=True)
