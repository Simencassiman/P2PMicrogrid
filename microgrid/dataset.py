# Python Libraries
from typing import List, Tuple
import re
from datetime import datetime, timedelta

import pandas as pd
import tensorflow as tf
import numpy as np


# Local modules
import setup
import database as db


# Define data splits
data_month = 10
testing_days = [8, 9, 10, 19, 20]
validation_days = [18]
training_days = list(range(11, 18))

start_day = min(min(testing_days), min(validation_days), min(training_days))
end_day = max(max(testing_days), max(validation_days), max(training_days))
start = datetime(2021, data_month, start_day)
end = datetime(2021, data_month, end_day) + timedelta(days=1)   # Last day is not included, so add 1 day

# Define columns with relevant information, used to select from dataframe
env_cols = ['day', 'time', 'temperature']
agent_cols = ['pv']
load_cols = ['l0', 'l1', 'l2', 'l3', 'l4']
cols = env_cols + agent_cols + load_cols


def compute_time_slot(time: str) -> int:
    t = datetime.strptime(time, '%H:%M:%S')

    return (t.minute / setup.TIME_SLOT) + t.hour * setup.MINUTES_PER_HOUR / setup.TIME_SLOT


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    # Extract and normalize timeslot
    df['time'] = df['time'].map(lambda t: compute_time_slot(t))
    df['time'] = df['time'] / 96.

    # Normalize power
    for load_col in load_cols:
        df[load_col] = (df[load_col].astype(float) / df[load_col].max().astype(float))
    df['pv'] = (df['pv'].astype(float) / df['pv'].max().astype(float))

    # Select relevant columns
    new_df = df[cols]

    return new_df


def get_data_from_csv() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv('data_env.csv', index_col=0), pd.read_csv('data_agent.csv', index_col=0)


def get_data(days: List[int]) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    con = db.get_connection()

    try:
        # Get data from local database
        df = db.get_data(con, start, end)
    finally:
        if con:
            con.close()

    # Only keep relevant days
    df['day'] = df['date'].map(lambda d: int(re.match(r'.*-([0-9]+)$', d).groups()[0]))
    df = df[df['day'].map(lambda d: d in days)]

    # Process data to match observation data for RL agents
    df = process_dataframe(df)

    agent_dfs = [df[[l] + agent_cols].rename(columns={l: 'load'}) for l in load_cols]

    return df[env_cols], agent_dfs


def get_train_data() -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    env_df, agent_dfs = get_data(training_days)
    env_df.drop(axis=1, labels='day', inplace=True)

    return env_df, agent_dfs


def get_validation_data() -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    return get_data(validation_days)


def get_test_data() -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    return get_data(testing_days)


def dataframe_to_dataset(df: pd.DataFrame, roll_len: int = -1, axis: int = 0) -> tf.data.Dataset:
    data = np.array(df, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((data, np.roll(data, roll_len, axis=axis)))

    return ds


### Run ###
if __name__ == '__main__':
    # Get a connection to the database

    env_df, agent_dfs = get_validation_data()

    print(env_df.head())
    for df in agent_dfs:
        print(df.head())
