import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Union, Optional
from functools import reduce

from microgrid import MINUTES_PER_HOUR
from config import DB_PATH, TIME_SLOT
from database import get_connection, get_data


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

conn = get_connection(DB_PATH)
start = datetime(2021, 11, 1)
val_start = datetime(2021, 11, 2)
val_end = datetime(2021, 11, 2)
end = datetime(2021, 11, 2)
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

    def __init__(self, train_df: Union[pd.DataFrame, List[pd.DataFrame]],
                 val_df: Union[pd.DataFrame, List[pd.DataFrame]],
                 test_df: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
                 input_width: int = 1, label_width: int = 1,
                 shift: int = 1, label_columns: List[str] = None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        if isinstance(train_df, list):
            self.column_indices = {name: i for i, name in
                                   enumerate(train_df[0].columns)}
        else:
            self.column_indices = {name: i for i, name in
                                   enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    @property
    def train_ds(self):
        if isinstance(self.train_df, list):
            return reduce(lambda ds1, ds2: ds1.concatenate(ds2),
                          map(lambda df: self.make_dataset(df),
                              self.train_df))
        return self.make_dataset(self.train_df)

    @property
    def val_ds(self):
        if isinstance(self.val_df, list):
            return reduce(lambda ds1, ds2: ds1.concatenate(ds2),
                          map(lambda df: self.make_dataset(df),
                              self.val_df))
        return self.make_dataset(self.val_df)

    @property
    def test_ds(self):
        if self.test_df is None:
            return None
        elif isinstance(self.test_df, list):
            return reduce(lambda ds1, ds2: ds1.concatenate(ds2),
                          map(lambda df: self.make_dataset(df),
                              self.test_df))
        return self.make_dataset(self.test_df)

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
wg = WindowGenerator(train_df=train_df_1, val_df=None,
                     input_width=horizon, shift=horizon, label_width=horizon, label_columns=list(val_df.columns))
# X = np.sin(np.arange(0, 100, step=0.1))
# train_ds = TimeseriesGenerator(X[:, None], X[:, None], input_width=3, shift=3, batch_size=1)


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pre = keras.Sequential([
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(100, activation='relu')
        ])
        self.lstm = keras.layers.LSTM(100, return_sequences=True)
        self.post = keras.Sequential([
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(2, activation='linear')
        ])
        self._layers = keras.Sequential([
            self.pre,
            self.lstm,
            self.lstm,
            self.post
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._layers(x)


model = MyModel()
loss_object = tf.keras.losses.MeanSquaredError()
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    [190*93], [1e-4, 0.1e-5])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)      # learning_rate=lr_schedule

train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
val_loss = tf.keras.metrics.MeanSquaredError(name='validation_loss')


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(x)
        loss = loss_object(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(y, predictions)

    return loss

@tf.function
def test_step(x, y):
    predictions = model(x)
    val_loss.update_state(y, predictions)


EPOCHS = 200


def main() -> None:

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        val_loss.reset_states()

        print(f'Epoch {epoch + 1}')

        print('Training:')
        for x, y in tqdm(wg.train_ds):
            loss = train_step(x, y)

        print("Training Loss: {:e}".format(train_loss.result()))

        # print('Validating:')
        # for x, y in tqdm(wg.val_ds):
        #     test_step(x, y)
        #
        # print("Validation Loss: {:e}".format(val_loss.result()))

        print('-----------------------')

    # Visualiaze predictions
    len_test = 20

    predictions1 = []
    targets1 = []
    for i, (x, y) in enumerate(wg.train_ds):
        predictions1.append(model(x).numpy()[0, 0, :])
        targets1.append(y.numpy()[0, 0, :])

    targets1 = np.stack(targets1)
    predictions1 = np.stack(predictions1)

    plt.figure(1)
    plt.plot(np.arange(targets1.shape[0]), targets1, predictions1)
    plt.legend(['Target load', 'Target pv',
                'Prediction load', 'Prediction pv'])

    # predictions2 = np.zeros([len_test])
    # targets2 = np.zeros([len_test])
    # for i, (x, y) in enumerate(wg.val_ds.take(len_test)):
    #     predictions2[i] = model(x)[:, 0, :].numpy().reshape([-1])
    #     targets2[i] = y[:, 0, :].numpy().reshape([-1])
    #
    # plt.figure(2)
    # plt.plot(np.arange(targets2.shape[0]), targets2, predictions2)
    # plt.legend(['Target', 'Prediction'])
    plt.show()


if __name__ == '__main__':

    x = tf.Variable(2.0)
    y = tf.Variable(3.0)

    with tf.GradientTape() as t:
        y_sq = y ** 2
        z1 = x ** 2 + tf.identity(tf.stop_gradient(y_sq))
        z = z1 + 0.5 * y

    [dz_dx, dz_dy] = t.gradient(z, [x, y])

    print('dz/dx:', dz_dx)  # 2*x => 4
    print('dz/dy:', dz_dy)
