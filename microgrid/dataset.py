import pandas as pd
import tensorflow as tf
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Union, Optional
from functools import reduce

from config import DB_PATH, TIME_SLOT, MINUTES_PER_HOUR
from database import get_connection, get_data


conn = get_connection(DB_PATH)
start = datetime(2021, 11, 1)
val_start = datetime(2021, 11, 7)
val_end = datetime(2021, 11, 8)
end = datetime(2021, 12, 1)
df = get_data(conn, start, end)


def compute_time_slot(time) -> int:
    t = datetime.strptime(time, '%H:%M:%S')

    return (t.minute / TIME_SLOT) + t.hour * MINUTES_PER_HOUR / TIME_SLOT


df['time'] = df['time'].map(lambda t: compute_time_slot(t))
df[['year', 'month', 'day']] = df['date'].str.split(pat='-', expand=True)
df['time'] = df['time'] / 96.
df['day'] = df['day'].astype(float) / 31.
df['month'] = df['month'].astype(float) / 12.
df['temperature'] = df['temperature'].astype(float) / df['temperature'].max().astype(float)
df['l0'] = df['l0'].astype(float) / df['l0'].max().astype(float)
df['pv'] = df['pv'].astype(float) / df['pv'].max().astype(float)
df['date'] = df['date'].map(lambda t: datetime.strptime(t, '%Y-%m-%d'))

cols = ['time', 'day', 'month', 'temperature', 'cloud_cover', 'humidity', 'l0', 'pv']
train_df_1 = df[df['date'] < val_start][cols]
train_df_2 = df[df['date'] > val_end][cols]
val_df = df[(val_start <= df['date']) & (df['date'] <= val_end)][cols]


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


horizon = 3
nr_input_features = 8
nr_actions = 1

def split_ds(in_features, t_features):
    in_features.set_shape([None, 3, None])
    t_features.set_shape([None, 3, None])
    inputs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    targets = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    horizon = 2
    for i in tf.range(in_features.shape[1] - horizon + 1):
        inputs = inputs.write(i, in_features[:, i:i + horizon, :])
        targets = targets.write(i, t_features[:, i:i + horizon, :])

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(inputs.stack()),
        tf.data.Dataset.from_tensor_slices(targets.stack())
    ))
    return ds


if __name__ == '__main__':
    train_ds = WindowGenerator(df=[train_df_1, train_df_2],
                               input_width=horizon, label_width=horizon,
                               label_columns=list(val_df.columns)).days
    validation_ds = WindowGenerator(df=train_df_1,
                                    input_width=horizon, label_width=horizon,
                                    label_columns=list(val_df.columns)).test_ds

    data_spec = (
        tf.TensorSpec([horizon, nr_input_features], tf.float32, 'state'),
        tf.TensorSpec([horizon, nr_actions], tf.float32, 'action'),
        tf.TensorSpec([1], tf.float32, 'reward'),
        tf.TensorSpec([horizon, nr_input_features], tf.float32, 'next_state')
    )

    batch_size = 4
    max_length = 5000

    replay_buffer = TFUniformReplayBuffer(
        data_spec,
        batch_size=batch_size,
        max_length=max_length
    )

    for day in train_ds.take(1):
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        next_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for t, (x, y) in enumerate(day.take(4)):
            print(x)
            states = states.write(t, x)
            actions = actions.write(t, x[:, :, -1:])
            rewards = rewards.write(t, tf.expand_dims(tf.math.reduce_sum(x[:, :, -1] - y[:, :, -1], axis=-1), axis=0))
            next_states = next_states.write(t, y)

        replay_buffer.add_batch((
            states.concat(),
            actions.concat(),
            rewards.concat(),
            next_states.concat()
        ))

    for i in range(3):
        print('-----')
        for (s, a, r, ns), _ in iter(replay_buffer.as_dataset(sample_batch_size=2).take(2)):
            print(s)
