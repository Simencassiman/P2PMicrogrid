# Python Libraries
import sqlite3
import os.path as osp
import pandas as pd
from datetime import datetime

# Local modules
from config import DATA_PATH, DB_FILE
import access_smarthor_data_api as api


def get_connection(file=DB_FILE) -> sqlite3.Connection:

    conn = None
    try:
        conn = sqlite3.connect(osp.join(DATA_PATH, file))
        return conn
    except Exception as e:
        print(e)

    return conn


def create_tables(cursor: sqlite3.Cursor) -> None:

    if cursor is not None:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS environment
            (date text NOT NULL, time text NOT NULL, utc text NOT NULL, 
            temperature real, cloud_cover real, humidity real, irradiation real, pv real,
            PRIMARY KEY (date, time, utc) )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS load
            (date text NOT NULL, time text NOT NULL, utc text NOT NULL, 
            load_0 real,
            PRIMARY KEY (date, time, utc) )
        """)

    else:
        print('Unable to create tables')


def insert_data_from_dict(cur: sqlite3.Cursor, df: pd.DataFrame) -> None:

    df_l = df[['date', 'time', 'utc', 'load']]

    env_records = [*zip(df['date'], df['time'], df['utc'], df['temperature'], df['cloud_cover'], df['humidity'],
                        [0.0] * len(df), df['pv'])]
    load_records = [row for _, row in df_l.iterrows()]

    cur.executemany("INSERT INTO environment VALUES (?,?,?,?,?,?,?,?)", env_records)
    cur.executemany("INSERT INTO load VALUES (?,?,?,?)", load_records)


def get_data(con: sqlite3.Connection, start: datetime, end: datetime) -> pd.DataFrame:

    query_env = """
        SELECT * 
        FROM environment 
        WHERE date >= ? AND date < ?     
    """

    query_l = """
        SELECT * 
        FROM load 
        WHERE date >= ? AND date < ?     
    """

    df_env = pd.read_sql_query(query_env, con, params=(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    df_l = pd.read_sql_query(query_l, con, params=(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))

    df = pd.merge(df_env, df_l, on=['date', 'time', 'utc'], copy=False)

    return df


if __name__ == '__main__':

    start = datetime(2021, 10, 1)
    end = datetime(2021, 11, 1)

    conn = get_connection()

    if conn is not None:
        cursor = conn.cursor()

        df = get_data(conn, start, end)

        print(df.head())
        print(df.tail())

    else:
        print('Could not connect to database')
