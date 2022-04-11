from typing import List
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import collections
import statistics
import gc

import rl


class ActorModel(keras.Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.pre = keras.Sequential([
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(100, activation='relu')
        ])
        self.lstm = keras.layers.LSTM(100, return_sequences=True)
        self.post = keras.Sequential([
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self._layers = keras.Sequential([
            self.pre,
            self.lstm,
            self.lstm,
            self.post
        ])

        self.concat = keras.layers.Concatenate()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._layers(x)


class CriticModel(keras.Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.pre = keras.Sequential([
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(100, activation='relu')
        ])
        self.lstm = keras.layers.LSTM(100, return_sequences=True)
        self.post = keras.Sequential([
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(1)
        ])
        self._layers = keras.Sequential([
            self.pre,
            self.lstm,
            self.lstm,
            self.post
        ])

        self.concat = keras.layers.Concatenate()

    def call(self, x: List[tf.Tensor]) -> tf.Tensor:
        return tf.math.reduce_sum(self._layers(self.concat(x)), axis=-2)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.07, theta=.02, dt=1e-2, sd: float = 0.2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self._sd = sd
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.random.normal(size=self.mu.shape, scale=self._sd)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


### Set up training ###
min_episodes_criterion = 100
begin_episodes = 0 * 1000
max_episodes = 2 * 1000

mse_loss = tf.keras.losses.MeanSquaredError()

bu = 10000
bs = 128
gamma = 0
lr = 1e-7
tau = 0.005
theta = 0.1
sigma = 0.1
sd = 1.0
activation = 'sigmoid'

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
ddpg = rl.DDPG(outputs=1, buffer_size=bu, batch_size=bs, gamma=gamma,
               critic_loss=mse_loss,
               actor_loss=mse_loss,
               critic_optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
               actor_optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
               tau=tau, theta=theta, sigma=sigma, sd=sd)

ddpg.actor = ActorModel()
ddpg.critic = CriticModel()
ddpg.target_actor = ActorModel()
ddpg.target_critic = CriticModel()


if __name__ == '__main__':

    # rl.load_models('tests', 'gamma=0;bs=32;ls=1e-06', 0, ddpg.actor, ddpg.critic, ddpg.target_actor, ddpg.target_critic)

    with trange(begin_episodes, max_episodes) as episodes:
        for episode in episodes:

            result = ddpg.train_episode()

            # Log progress
            episodes_reward.append(result)

            # Show average episode reward every x episodes
            if episode % min_episodes_criterion == 0:
                # Compute statistics
                training = statistics.mean(episodes_reward)
                validation = float(tf.math.reduce_sum(rl.test(ddpg.actor)).numpy())
                q_error = []
                for x, y in rl.wg.train_ds:
                    actions = ddpg.actor(x)
                    q_vals = ddpg.critic([x, actions])
                    error = -tf.math.reduce_sum(tf.math.squared_difference(y[:, :, -1:], actions), axis=-2)
                    q_error.append(tf.math.reduce_sum(tf.math.squared_difference(error, q_vals)).numpy())

                # Report results
                print(f'Episode {episode}: running reward: {training:.3f}, validation: {validation:.3f}, '
                      f'Q-error: {statistics.mean(q_error):.3f}')

            gc.collect()

    rl.save_models('tests', 'gamma=0;bs=128;ls=1e-07', 0, ddpg.actor, ddpg.critic, ddpg.target_actor, ddpg.target_critic)

    # Visualize predictions
    predictions = []
    targets = []
    errors = []
    q_estimates = []
    for x, y in rl.wg.train_ds:
        actions = ddpg.actor(x)
        q_vals = ddpg.critic([x, x[:, :, -1:]])

        # for target, prediction in tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x),
        #                                                tf.data.Dataset.from_tensor_slices(y))):
        #     targets.append(target[-1, -1:])
        #     predictions.append(prediction[-1, -1:])

        for target, prediction, a, q in tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(y),
                                                             tf.data.Dataset.from_tensor_slices(x[:, :, -1:]),
                                                             tf.data.Dataset.from_tensor_slices(actions),
                                                             tf.data.Dataset.from_tensor_slices(q_vals))):
            predictions.append(a[-1, :].numpy())
            targets.append(target[-1, -1:].numpy())
            errors.append(-tf.math.reduce_sum(tf.math.squared_difference(target[:, -1:], prediction)))
            q_estimates.append(q)

    targets = np.stack(targets)
    predictions = np.stack(predictions)
    errors = np.stack(errors)
    q_estimates = np.stack(q_estimates)

    plt.figure(1)
    plt.plot(np.arange(targets.shape[0]), targets, predictions)
    plt.legend(['Target load',
                'Prediction load'])

    plt.figure(2)
    plt.plot(np.arange(targets.shape[0]), errors, q_estimates)
    plt.legend(['Error',
                'Q'])
    plt.show()
