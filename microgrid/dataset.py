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
data_month = 9
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


class WindowGenerator:
    # Code taken from: https://www.tensorflow.org/tutorials/structured_data/time_series

    def __init__(self,
                 df: Union[pd.DataFrame, List[pd.DataFrame]],
                 input_width: int = 1,
                 label_width: int = 1,
                 shift: int = 1,
                 day_size: int = 96,
                 label_columns: List[str] = None):
        # Store the raw data.
        self.df = df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        if isinstance(df, list):
            self.column_indices = {name: i for i, name in
                                   enumerate(df[0].columns)}
        else:
            self.column_indices = {name: i for i, name in
                                   enumerate(df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        self.day_size = day_size

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    @property
    def days(self):
        if isinstance(self.df, list):
            return reduce(lambda ds1, ds2: ds1.concatenate(ds2),
                          map(lambda df: self.daily_dataset(df),
                              self.df))
        return self.daily_dataset(self.df)

    @property
    def test_ds(self):
        if isinstance(self.df, list):
            return reduce(lambda ds1, ds2: ds1.concatenate(ds2),
                          map(lambda df: self.make_dataset(np.array(df, dtype=np.float32)),
                              self.df))
        return self.make_dataset(np.array(self.df, dtype=np.float32))

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        # labels = labels[:, :, -2:]

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)

        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=1)

        ds = ds.map(self.split_window)

        return ds

    def split_day(self, in_features, t_features):
        in_features.set_shape([None, self.day_size, None])
        t_features.set_shape([None, self.day_size, None])

        inputs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        targets = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for i in tf.range(in_features.shape[1] - self.input_width + 1):
            inputs = inputs.write(i, in_features[:, i:i + self.input_width, :])
            targets = targets.write(i, t_features[:, i:i + self.input_width, :])

        ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(inputs.stack()),
            tf.data.Dataset.from_tensor_slices(targets.stack())
        ))
        return ds

    def daily_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        inputs = data[:-self.shift, :]
        targets = data[self.shift:, :]

        ds = tf.data.Dataset.zip((
            tf.keras.utils.timeseries_dataset_from_array(
                data=inputs,
                targets=None,
                sequence_length=self.day_size,
                sequence_stride=1,
                shuffle=False,
                batch_size=1),
            tf.keras.utils.timeseries_dataset_from_array(
                data=targets,
                targets=None,
                sequence_length=self.day_size,
                sequence_stride=1,
                shuffle=False,
                batch_size=1)
        )).shuffle(data.shape[1])

        ds = ds.map(self.split_day)

        return ds

    def plot(self, model=None, plot_col='Load (W)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


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

    # Add price for each timeslot
    # new_df['price'] = (
    #         (cf.GRID_COST_AVG
    #          + cf.GRID_COST_AMPLITUDE
    #          * np.sin(2 * np.pi * np.array(df['time'])
    #                   * MINUTES_PER_HOUR / TIME_SLOT * HOURS_PER_DAY / cf.GRID_COST_PERIOD + cf.GRID_COST_PHASE)
    #          ) / CENTS_PER_EURO  # from c€ to €
    # )

    return new_df


def get_data_from_csv() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv('data_env.csv', index_col=0), pd.read_csv('data_agent.csv', index_col=0)


def get_data(days: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Get data from local database
    df = db.get_data(conn, start, end)

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


# def split_ds(in_features, t_features):
#     in_features.set_shape([None, 3, None])
#     t_features.set_shape([None, 3, None])
#     inputs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#     targets = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#
#     horizon = 2
#     for i in tf.range(in_features.shape[1] - horizon + 1):
#         inputs = inputs.write(i, in_features[:, i:i + horizon, :])
#         targets = targets.write(i, t_features[:, i:i + horizon, :])
#
#     ds = tf.data.Dataset.zip((
#         tf.data.Dataset.from_tensor_slices(inputs.stack()),
#         tf.data.Dataset.from_tensor_slices(targets.stack())
#     ))
#     return ds


horizon = 3
nr_input_features = 8
nr_actions = 1


if __name__ == '__main__':
    env_df, aget_df = get_validation_data()
    print(env_df.head())

    env_df.to_csv('../data/data_env_validation.csv')
    aget_df.to_csv('../data/data_agent_validation.csv')
