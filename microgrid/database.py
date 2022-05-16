# Python Libraries
import sqlite3
import os.path as osp
from typing import List, Union
import matplotlib.pyplot as plt
import re

import numpy as np
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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_results 
            (setting text NOT NULL, implementation text NOT NULL, agent integer NOT NULL, day integer NOT NULL,
             time real NOT NULL, load real, pv real, temperature real, heatpump real, cost real, 
            PRIMARY KEY (setting, implementation, agent, day, time) )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results
            (setting text NOT NULL, implementation text NOT NULL, agent integer NOT NULL, day integer NOT NULL, 
            time real NOT NULL, load real, pv real, temperature real, heatpump real, cost real,
            PRIMARY KEY (setting, implementation, agent, day, time) )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rounds_comparison  
            (setting text NOT NULL, agent integer NOT NULL, day integer NOT NULL, time real NOT NULL,
             round integer NOT NULL, decision real,
            PRIMARY KEY (setting, agent, day, time, round))
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


def generate_additional_load(con: sqlite3) -> None:
    if con is not None:
        cursor = conn.cursor()

        query_1 = """
                        SELECT *
                        FROM load
                        WHERE date LIKE '%-10-%'
                    """
        query_2 = """
                        UPDATE load
                        SET l4 = ?
                        WHERE date = ? AND time = ? AND utc = ?
                    """

        df = pd.read_sql_query(query_1, conn)

        df.loc[df['l0'] > df['l0'].median() * 2, 'l0'] = df['l0'].median() * 2
        max_l = df['l0'].max()
        df['l0'] = 1 - df['l0'] / max_l
        df['days'] = df['date'].map(lambda d: int(re.match(r'.*-([0-9]{2})$', d).groups()[0]))
        days = df['days'].unique().tolist()
        df2 = pd.concat(map(lambda d: df[df['days'] == d], np.random.permutation(days))).reset_index()
        df['l1'] = df2['l0'] * max_l

        records = [*zip(df['l1'], df['date'], df['time'], df['utc'])]

        cursor.executemany(query_2, records)

        conn.commit()


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
                 training: float, validation: float, q_error: float) -> None:
    if con is not None:
        cursor = con.cursor()

        try:
            query = "INSERT INTO hyperparameters_single_day VALUES (?,?,?,?,?,?)"

            cursor.execute(query, (settings, trial, episode, training, validation, q_error))

            con.commit()
        finally:
            if cursor:
                cursor.close()


def log_predictions(con: sqlite3.Connection, settings: str, date: List[float], time: List[float],
                    load: List[float], pv: List[float], target_load: List[float], target_pv: List[float]) -> None:
    if con is not None:
        cursor = con.cursor()

        try:
            query = "INSERT INTO single_day_best_results VALUES (?,?,?,?,?,?,?)"

            time = list(map(lambda t: str(t), time))
            n = len(load)
            records = [*zip([settings] * n, date, time, load, pv, target_load, target_pv)]

            cursor.executemany(query, records)

            con.commit()
        finally:
            if cursor:
                cursor.close()


def log_training_progress(con: sqlite3.Connection,
                          setting: str, agent_type: str, episode: int, reward: float, error: float) -> None:
    if con is not None:
        cursor = con.cursor()

        try:
            query = "INSERT INTO training_progress VALUES (?,?,?,?,?)"

            cursor.execute(query, (setting, agent_type, episode, reward, error))

            con.commit()
        finally:
            if cursor:
                cursor.close()


def get_training_progress(con: sqlite3.Connection) -> Union[pd.DataFrame, None]:
    if con:
        query = """
            SELECT *
            FROM training_progress
        """

        df = pd.read_sql_query(query, con)

        return df

    return None


def log_validation_results(con: sqlite3.Connection, setting: str, agent_id: int, days: List[int],
                           time: List[float], load: List[float], pv: List[float], temperature: List[float],
                           heatpump: List[float], cost: List[float], implementation: str) -> None:
    if con is not None:
        cursor = con.cursor()

        try:
            query = "INSERT INTO validation_results VALUES (?,?,?,?,?,?,?,?,?,?)"

            n = len(load)
            records = [*zip([setting] * n, [implementation] * n, [agent_id] * n, days, time, load, pv,
                            temperature, heatpump, cost)]

            cursor.executemany(query, records)

            con.commit()
        finally:
            if cursor:
                cursor.close()


def get_validation_results(con: sqlite3.Connection) -> Union[pd.DataFrame, None]:
    if con:
        query = """
            SELECT *
            FROM validation_results
        """

        df = pd.read_sql_query(query, con)

        return df

    return None


def log_test_results(con: sqlite3.Connection, setting: str, agent_id: int, days: List[int],
                           time: List[float], load: List[float], pv: List[float], temperature: List[float],
                           heatpump: List[float], cost: List[float], implementation: str) -> None:
    if con is not None:
        cursor = con.cursor()

        try:
            query = "INSERT INTO test_results VALUES (?,?,?,?,?,?,?,?,?,?)"

            n = len(load)
            records = [*zip([setting] * n, [implementation] * n, [agent_id] * n, days, time, load, pv,
                            temperature, heatpump, cost)]

            cursor.executemany(query, records)

            con.commit()
        finally:
            if cursor:
                cursor.close()


def get_test_results(con: sqlite3.Connection) -> Union[pd.DataFrame, None]:
    if con:
        query = """
            SELECT *
            FROM test_results
        """

        df = pd.read_sql_query(query, con)

        return df

    return None


def log_rounds_decision(con: sqlite3.Connection, setting: str, agent: int, days: List[int],
                        time: List[float], round: int, decisions: List[float]) -> None:
    if con:
        cursor = con.cursor()

        try:
            query = "INSERT INTO rounds_comparison VALUES (?,?,?,?,?,?)"

            nr_entries = len(time)
            data = [*zip([setting] * nr_entries, [agent] * nr_entries, days, time, [round] * nr_entries, decisions)]

            cursor.executemany(query, data)
            con.commit()

        finally:
            if cursor:
                cursor.close()


def get_rounds_decisions(con: sqlite3.Connection) -> Union[pd.DataFrame, None]:
    if con:
        query = """
            SELECT * 
            FROM rounds_comparison
        """
        return pd.read_sql_query(query, con)

    return None


if __name__ == '__main__':

    conn = get_connection()

    if conn is not None:

        cursor = conn.cursor()
        try:

            query = """
                SELECT * 
                FROM test_results
                WHERE setting LIKE '%pv%'
            """

            df = pd.read_sql_query(query, conn)
            print(df)

            # cursor.execute(query)
            # conn.commit()

        finally:
            cursor.close()
            conn.close()

    else:
        print('Could not connect to database')
