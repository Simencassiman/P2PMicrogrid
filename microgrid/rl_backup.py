import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, List
import collections
import statistics
import random
import sys

from microgrid import MINUTES_PER_HOUR
from config import DB_PATH, TIME_SLOT
from database import get_connection, get_data
from ml import WindowGenerator


### Parameter setup ###
seed = 42
# env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

conn = get_connection(DB_PATH)
start = datetime(2021, 11, 1)
val_start = datetime(2021, 11, 2)
val_end = datetime(2021, 11, 2)
end = datetime(2021, 11, 2)
df = get_data(conn, start, end)


### Prepare data ###
def compute_time_slot(time) -> int:
    t = datetime.strptime(time, '%H:%M:%S')

    return (t.minute / TIME_SLOT) + t.hour * MINUTES_PER_HOUR / TIME_SLOT


df['time'] = df['time'].map(lambda t: compute_time_slot(t))
df[['year', 'month', 'day']] = df['date'].str.split(pat='-', expand=True)
df['time'] = df['time'] / 96.
df['day'] = df['day'].astype(float) / 31.
df['month'] = df['month'].astype(float) / 12.
df['l0'] = df['l0'].astype(float) / df['l0'].max().astype(float)
df['pv'] = df['pv'].astype(float) / df['pv'].max().astype(float)
df['date'] = df['date'].map(lambda t: datetime.strptime(t, '%Y-%m-%d'))

train_df_1 = df[df['date'] < val_start][['time', 'day', 'month', 'l0', 'pv']]
train_df_2 = df[df['date'] > val_end][['time', 'day', 'month', 'l0', 'pv']]
val_df = df[(val_start <= df['date']) & (df['date'] <= val_end)][['time', 'day', 'month', 'l0', 'pv']]

horizon = 3
wg = WindowGenerator(train_df=train_df_1, val_df=None,
                     input_width=horizon, label_width=horizon, label_columns=list(val_df.columns))


### Create model ###
class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = collections.deque()

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm = keras.layers.LSTM(100, return_sequences=True)
        self.common = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(100, activation='relu'),
            self.lstm,
            self.lstm,
            keras.layers.Dense(32, activation='relu')
        ])
        self.actor = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(2)
        ])
        self.concat = keras.layers.Concatenate()
        self.critic = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])

        self.common_weights = None
        self.actor_weights = None
        self.critic_weights = None

        self.state_a: tf.Tensor = None
        self.state_c: tf.Tensor = None
        self.action: tf.Tensor = None
        self.q: tf.Tensor = None

    def build(self, *args, **kwargs):
        super(MyModel, self).build(*args, **kwargs)
        self.common_weights = self.common.trainable_weights
        self.actor_weights = self.actor.trainable_weights
        self.critic_weights = self.critic.trainable_weights

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Store relevant checkpoints to compute gradients later
        self.state_c = self.common(inputs)
        self.state_a = tf.identity(tf.stop_gradient(self.state_c))
        self.action = self.actor(self.state_a)
        self.q = tf.math.reduce_sum(self.critic(self.concat([self.state_c, self.action])))  # This should be one value
        return self.action, self.q

    def gradients(self, loss, tape: tf.GradientTape) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Gradient for critic section: loss wrt critic weights
        [dl_dwc, dl_ds] = tape.gradient(loss, [self.critic_weights, self.state_c])

        # Gradient for actor section: Q value with respect to action x action wrt actor weights
        [dq_dwa, dq_ds] = tape.gradient(self.q, [self.actor_weights, self.state_a])

        # Gradient for common part: (loss wrt critic s + loss wrt actor s) * s wrt weights
        ds_dwcm = tape.gradient(self.state, self.common_weights)
        dl_dwcm = tf.matmul((self.crititc_factor * dl_ds + self.actor_factor * dq_ds), ds_dwcm)

        return dq_dwa, dl_dwc, dl_dwcm

    @property
    def weights(self) -> Tuple:
        return self.actor_weights, self.critic_weights, self.common_weights


model = MyModel()


### Set up training procedure ###
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    state, reward, done, _ = (1, 2, 3, 4)   # env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action],
                             [tf.float32, tf.int32, tf.int32])

def run_episode(initial_state: tf.Tensor,
                model: tf.keras.Model,
                max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    # initial_state_shape = initial_state.shape
    # state = initial_state

    for t, (state, y) in enumerate(wg.train_ds):
        # Convert state into a batched tensor (batch size = 1)
        # state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action, value = model(state)

        # Store critic values
        values = values.write(int(t), tf.squeeze(value))

        # Store chosen action
        actions = actions.write(int(t), action)

        # Apply action to the environment to get next state and reward
        # state, reward, done = tf_env_step(action)
        # state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(int(t), float(-tf.math.reduce_sum(tf.math.squared_difference(y, action))))
        # if tf.cast(done, tf.bool):
        #     break
        # else:
        #     rewards = rewards.write(t, reward)

    actions = actions.stack()
    values = values.stack()
    rewards = rewards.stack()

    return actions, values, rewards

def get_expected_return(rewards: tf.Tensor,
                        gamma: float,
                        standardize: bool = False) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)

    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))

    return returns


### Set up loss function ###
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(action_probs: tf.Tensor,
                 values: tf.Tensor,
                 returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""

    # advantage = returns - values
    #
    # action_log_probs = tf.math.log(action_probs)

    critic_loss = huber_loss(values, returns)

    return critic_loss


### Set up training step ###
# Learning rate for actor-critic models
# critic_lr = 0.002
# actor_lr = 0.001
#
# critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
# actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""

    with tf.GradientTape(persistent=True) as tape:

        # Run the model for one episode to collect training data
        # action, values, rewards = run_episode(initial_state, model, max_steps_per_episode)
        actions, values, rewards = run_episode(None, model, None)

        # Calculate expected returns
        returns = get_expected_return(rewards, gamma)

        # Convert training data to appropriate TF tensor shapes
        actions, values, returns = [
            tf.expand_dims(x, 1) for x in [actions, values, returns]]

        # Calculating loss values to update our network
        loss = compute_loss(actions, values, returns)

    for g, w in zip(model.gradients(loss, tape), model.weights):
        optimizer.apply_gradients(zip(g, w))

    # # Compute the gradients from the loss
    # grads = tape.gradient(loss, model.trainable_variables)
    #
    # # Apply the gradients to the model's parameters
    # optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
val_loss = tf.keras.metrics.MeanSquaredError(name='validation_loss')

min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 1000

# Discount factor for future rewards
gamma = 0.99

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

# actor_model = get_actor()
# critic_model = get_critic()
#
# target_actor = get_actor()
# target_critic = get_critic()
#
# # Making the weights equal initially
# target_actor.set_weights(actor_model.get_weights())
# target_critic.set_weights(critic_model.get_weights())


# Used to update target networks
# tau = 0.005
#
# buffer = Buffer(50000, 64)


if __name__ == '__main__':

    with trange(max_episodes) as t:
        for i in t:
            # initial_state = tf.constant(env.reset(), dtype=tf.float32)
            # env.reset()
            # episode_reward = int(train_step(
            #     initial_state, model, optimizer, gamma, max_steps_per_episode))

            # Run the model for one episode to collect training data
            # action, values, rewards = run_episode(initial_state, model, max_steps_per_episode)
            actions, values, rewards = run_episode(None, model, None)

            # Calculate expected returns
            returns = get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            actions, values, returns = [
                tf.expand_dims(x, 1) for x in [actions, values, returns]]

            # Run through episode
            # Store in buffer
            # Train per step in buffer

            episode_reward = int(train_step(None, model, optimizer, gamma, None))

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            t.set_description(f'Episode {i}')
            t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

            # Show average episode reward every 10 episodes
            if i % 10 == 0:
                pass # print(f'Episode {i}: average reward: {avg_reward}')

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

    # Visualiaze predictions
    # len_test = 20
    #
    # predictions1 = []
    # targets1 = []
    # for i, (x, y) in enumerate(wg.train_ds):
    #     predictions1.append(model(x).numpy()[0, 0, :])
    #     targets1.append(y.numpy()[0, 0, :])
    #
    # targets1 = np.stack(targets1)
    # predictions1 = np.stack(predictions1)
    #
    # plt.figure(1)
    # plt.plot(np.arange(targets1.shape[0]), targets1, predictions1)
    # plt.legend(['Target load', 'Target pv',
    #             'Prediction load', 'Prediction pv'])
    # plt.show()
