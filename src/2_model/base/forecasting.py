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


def get_data(use_cache, output_dir):
    """Load data"""

    if use_cache:
        with open(os.path.join(output_dir, 'region_demand.pickle'), 'rb') as f:
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
        sql = f"""SELECT SETTLEMENTDATE, REGIONID, TOTALDEMAND FROM dispatchregionsum"""

        # Execute query and return data
        cur.execute(sql)
        res = cur.fetchall()

        # Convert to DataFrame
        data = pd.DataFrame(res)

        # Save demand data
        with open(os.path.join(output_dir, 'region_demand.pickle'), 'wb') as f:
            pickle.dump(data, f)

    return data


def reindex_demand_data(df):
    """Pivot DataFrame and re-index to different time resolutions"""

    # Pivot so regions are columns, and index is timestamp
    df_p = (df.drop_duplicates(subset=['SETTLEMENTDATE', 'REGIONID'], keep='last')
            .pivot(index='SETTLEMENTDATE', columns='REGIONID', values='TOTALDEMAND').astype(float))

    # Regional energy demand at hourly resolution (5 min intervals so MWh = MW x (1/12))
    df_h = df_p.mul(1/12).resample('1h', label='right', closed='right').sum()

    # Regional energy demand at daily resolution
    df_d = df_h.resample('1d', closed='right').sum()

    # Get ISO calendar dates and add them to the DataFrame
    iso_dates = df_d.apply(lambda x: pd.Series({k: v for k, v in zip(('year', 'week', 'day'), x.name.isocalendar())}),
                           axis=1)
    df_dd = pd.concat([df_d, iso_dates], axis=1)

    # Weekly energy demand. Removing first and last weeks because they are not comprised of a total of seven days.
    df_w = df_dd.groupby(['year', 'week']).sum().drop(index=[(2013, 1), (2019, 1)]).drop('day', axis=1)

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


if __name__ == '__main__':
    # Output directory
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output')

    # Load demand data
    demand = get_data(use_cache=True, output_dir=output_directory)

    # Pre-process demand data
    pivoted, hourly, daily, weekly = reindex_demand_data(demand)

    # Dataset
    dataset = construct_dataset(weekly, past_lags=12, future_intervals=6)

    # Split data into training and testing sets
    x_train, y_train, x_test, y_test = train_test_split(dataset, train_proportion=0.7)

    model = sm.QuantReg(y_train['future_1'], x_train)
    res = model.fit(q=0.5)

    models = {}
    for q in np.linspace(0.05, 0.95, 10):
        models[round(q, 2)] = {}
        for i in range(1, 7):
            models[round(q, 2)][i] = sm.QuantReg(y_train[f'future_{i}'], x_train).fit(q=q)

    prediction = {}
    series = x_test.iloc[0].to_frame().T
    series_y = y_test.iloc[0]
    for q in np.linspace(0.05, 0.95, 10):
        prediction[round(q, 2)] = {}
        for i in range(1, 7):
            prediction[round(q, 2)][i] = models[round(q, 2)][i].predict(series).values[0]

    fig, ax = plt.subplots()
    for q in prediction.keys():
        ax.plot(list(prediction[q].values()))

    pairs = [(round(0.95 - 0.1*i, 2), round(0.05 + 0.1*i, 2)) for i in range(0, 5)]

    for q1, q2 in pairs:
        ax.fill_between([i for i in range(0, 6)], list(prediction[q1].values()), list(prediction[q2].values()),
                        alpha=0.2, color='blue')
    plt.show()

    series_y.reset_index().drop('index', axis=1).plot(ax=ax, color='r')
    plt.show()

