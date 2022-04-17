# Python Libraries
from typing import List, Tuple, Union
from functools import reduce
import re
from datetime import datetime, timedelta

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Local modules
import config as cf
from config import TIME_SLOT, MINUTES_PER_HOUR, CENTS_PER_EURO, HOURS_PER_DAY
import database as db
from database import get_connection


# Get a connection to the database
conn = get_connection(cf.DB_PATH)

# Define data splits
data_month = 10
testing_days = [10]
validation_days = [18]
training_days = list(range(11, 18))

start_day = min(min(testing_days), min(validation_days), min(training_days))
end_day = max(max(testing_days), max(validation_days), max(training_days))
start = datetime(2021, data_month, start_day)
end = datetime(2021, data_month, end_day) + timedelta(days=1)   # Last day is not included, so add 1 day

# Define columns with relevant information, used to select from dataframe
env_cols = ['time', 'temperature']
agent_cols = ['l0', 'pv']
cols = env_cols + agent_cols


def compute_time_slot(time: str) -> int:
    t = datetime.strptime(time, '%H:%M:%S')

    return (t.minute / TIME_SLOT) + t.hour * MINUTES_PER_HOUR / TIME_SLOT


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    # Extract and normalize timeslot
    df['time'] = df['time'].map(lambda t: compute_time_slot(t))
    df['time'] = df['time'] / 96.

    # Normalize power
    df['l0'] = (df['l0'].astype(float) / df['l0'].max().astype(float))
    df['pv'] = (df['pv'].astype(float) / df['pv'].max().astype(float))

    # Select relevant columns
    new_df = df[cols]

    return new_df


def get_data_from_csv() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv('data_env.csv', index_col=0), pd.read_csv('data_agent.csv', index_col=0)


def get_data(days: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    con = db.get_connection()

    try:
        # Get data from local database
        df = db.get_data(con, start, end)
    except:
        pass
    finally:
        if con:
            con.close()

    # Only keep relevant days
    df = df[df['date'].map(lambda d: int(re.match(r'.*-([0-9]+)$', d).groups()[0]) in days)]

    # Process data to match observation data for RL agents
    df = process_dataframe(df)

    return df[env_cols], df[agent_cols]


def get_train_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return get_data(training_days)


def get_validation_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return get_data(validation_days)


def get_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return get_data(testing_days)


def dataframe_to_dataset(df: pd.DataFrame, roll_len: int = -1, axis: int = 0) -> tf.data.Dataset:
    data = np.array(df, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((data, np.roll(data, roll_len, axis=axis)))

    return ds


if __name__ == '__main__':
    env_df, agent_df = get_validation_data()
    print(env_df.head())

    env_df.to_csv('../data/data_env_validation.csv')
    agent_df.to_csv('../data/data_agent_validation.csv')
