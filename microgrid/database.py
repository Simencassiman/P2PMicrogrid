# Python Libraries
import sqlite3
import os.path as osp
import pandas as pd
from datetime import datetime

# Local modules
from config import data_path, db_file
from access_smarthor_data_api import get_data


def get_connection(file=db_file) -> sqlite3.Connection:

    conn = None
    try:
        conn = sqlite3.connect(osp.join(data_path, file))
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


if __name__ == '__main__':

    start = datetime(2021, 5, 1)
    end = datetime(2021, 12, 1)

    conn = get_connection()

    if conn is not None:
        cursor = conn.cursor()

        df = pd.read_sql_query("""
            SELECT * 
            FROM environment 
            WHERE date = '2021-11-01' AND time > '13:00:00'
        """, conn)

        print(df.head())

    else:
        print('Could not connect to database')
