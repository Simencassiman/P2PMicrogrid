# Python Libraries
import sqlite3
import os.path as osp
from typing import List
import matplotlib.pyplot as plt
import re

import pandas as pd
from datetime import datetime

# Local modules
import config
from config import DATA_PATH, DB_FILE, TIME_SLOT, MINUTES_PER_HOUR
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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hyperparameters_single_day 
            (settings text NOT NULL, trial integer NOT NULL, episode integer NOT NULL, 
            training real NOT NULL, validation real NOT NULL, 
            PRIMARY KEY (settings, trial, episode) )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS single_day_best_results 
            (settings text NOT NULL, date text NOT NULL, time text NOT NULL, load real, pv real, target_load real, 
            target_pv real,  
            PRIMARY KEY (settings, date, time) )
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


def get_load_data(con: sqlite3.Connection, start: datetime, end: datetime) -> pd.DataFrame:
    query = """
        SELECT * 
        FROM load 
        WHERE date >= ? AND date < ?     
    """

    return pd.read_sql_query(query, con, params=(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))


def log_training(con: sqlite3.Connection, settings: str, trial: int, episode: int,
                 training: float, validation: float) -> None:
    if con is not None:
        cursor = con.cursor()
        query = "INSERT INTO hyperparameters_single_day VALUES (?,?,?,?,?)"

        cursor.execute(query, (settings, trial, episode, training, validation))

        con.commit()


def log_predictions(con: sqlite3.Connection, settings: str, date: List[float], time: List[float],
                    load: List[float], pv: List[float], target_load: List[float], target_pv: List[float]) -> None:
    if con is not None:
        cursor = con.cursor()
        query = "INSERT INTO single_day_best_results VALUES (?,?,?,?,?,?,?)"

        time = list(map(lambda t: str(t), time))
        n = len(load)
        records = [*zip([settings] * n, date, time, load, pv, target_load, target_pv)]
        print(records[0])

        cursor.executemany(query, records)

        con.commit()


if __name__ == '__main__':

    # start = datetime(2021, 11, 1)
    # val_start = datetime(2021, 11, 15)
    # val_end = datetime(2021, 11, 21)
    # end = datetime(2021, 12, 1)

    conn = get_connection()

    if conn is not None:

        query = """
            SELECT settings, episode, AVG(training) as train
            FROM hyperparameters_single_day
            WHERE settings NOT LIKE '%ls=1e-05%'
            GROUP BY settings, episode 
            ORDER BY settings, episode 
        """

        df = pd.read_sql_query(query, conn)
        print(df.head())
        print(df.columns)
        pattern = r"(bs=.+?ls=1e-0[0-9]{1})"
        df['settings'] = df['settings'].apply(lambda x: re.search(pattern, x).group())
        settings = df['settings']
        df = df.pivot(index='episode', columns='settings', values='train')
        print(df.columns)

        ax = df.plot(xticks=df.index, ylabel='Reward')

        plt.show()

        # print(df['episode'].head())
        # print(df.columns)

        # df = get_load_data(conn, start, end)
        #
        # df['date'] = df['date'].map(lambda t: datetime.strptime(t, '%Y-%m-%d'))
        #
        # train_df = df[(df['date'] < val_start) | (df['date'] > val_end)]
        # val_df = df[(val_start <= df['date']) & (df['date'] <= val_end)]
        #
        # print(train_df.head())
        # print(train_df.tail())
        # print(train_df[train_df['date'] == datetime(2021,11,16)])
        # print(val_df.head())
        # print(val_df.tail())
        #
        # def compute_time_slot(time) -> int:
        #     t = datetime.strptime(time, '%H:%M:%S')
        #
        #     return (t.minute / TIME_SLOT) + t.hour * MINUTES_PER_HOUR / TIME_SLOT
        #
        # df['time'] = df['time'].map(lambda t: compute_time_slot(t))
        # df[['year', 'month', 'day']] = df['date'].str.split(pat='-', expand=True)
        # df.drop(['date', 'utc', 'year'], axis=1, inplace=True)
        # df = df[['time', 'day', 'month', 'l0']]
        # df['time'] = df['time'] / 96.
        # df['day'] = df['day'].astype(float) / 31.
        # df['month'] = df['month'].astype(float) / 12.
        # df['l0'] = df['l0'].astype(float) / df['l0'].max().astype(float)
        #
        # print(df.head())
        # print(df.tail())

    else:
        print('Could not connect to database')
