"""Demand probabilistic forecasting"""

import os

import MySQLdb
import MySQLdb.cursors
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    # Database parameters
    host = os.environ['LOCAL_HOST']
    schema = os.environ['LOCAL_SCHEMA']
    user = os.environ['LOCAL_USER']
    password = os.environ['LOCAL_PASSWORD']

    conn = MySQLdb.connect(host=host, db=schema, user=user, passwd=password, cursorclass=MySQLdb.cursors.DictCursor)
    cur = conn.cursor()
    sql = f"""SELECT SETTLEMENTDATE, REGIONID, TOTALDEMAND FROM dispatchregionsum"""

    cur.execute(sql)
    res = cur.fetchall()

    # Convert to DataFrame
    df = pd.DataFrame(res)

    # Pivot so regions are columns, and index is timestamp
    df_p = (df.drop_duplicates(subset=['SETTLEMENTDATE', 'REGIONID'], keep='last')
            .pivot(index='SETTLEMENTDATE', columns='REGIONID', values='TOTALDEMAND'))

