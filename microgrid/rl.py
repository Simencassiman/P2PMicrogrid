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
import random

from microgrid import MINUTES_PER_HOUR
from config import DB_PATH, TIME_SLOT
from database import get_connection, get_data
from ml import WindowGenerator

"""
Code adapted from
https://github.com/tensorflow/agents/blob/v0.12.0/tf_agents/agents/ddpg/ddpg_agent.py
https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html#References

Other sources
Target networks: https://www.nature.com/articles/nature14236.pdf
"""


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
class ReplayBuffer:

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.count = 0
        self.buffer = collections.deque()

    def add(self, s: tf.Tensor, a: tf.Tensor, r: tf.Tensor, ns: tf.Tensor) -> None:
        for i in tf.range(tf.shape(s)[0]):
            experience = (s[i, :, :], a[i, :, :], r[i], ns[i, :, :])

            if self.count < self.buffer_size:
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        """
        batch = []

        if self.count < self.batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, self.batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        ns_batch = np.array([_[3] for _ in batch])

        return s_batch, a_batch, r_batch, ns_batch

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

    def build(self, *args, **kwargs):
        super(MyModel, self).build(*args, **kwargs)
        self.common_weights = self.common.trainable_weights
        self.actor_weights = self.actor.trainable_weights
        self.critic_weights = self.critic.trainable_weights

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Store relevant checkpoints to compute gradients later
        state = self.common(inputs)
        action = self.actor(state)
        q = tf.math.reduce_sum(self.critic(self.concat([state, action])), axis=-2)  # This should be one value
        return action, q

    def predict(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Incrementally build up states as the model predicts actions
        # Return everything to allow learning from buffer
        return self(inputs)     # For now next state is fully determined

    def forward_critic(self, state: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        state_rep = self.common(state)
        critic_input = self.concat([state_rep, action])
        q = tf.math.reduce_sum(self.critic(critic_input), axis=-2)  # This should be one value
        return q

    def forward_actor(self, state: tf.Tensor):
        state_rep = self.common(state)
        action = self.actor(state_rep)
        critic_state = tf.identity(tf.stop_gradient(state_rep))
        critic_input = self.concat([critic_state, action])
        q = tf.math.reduce_sum(self.critic(critic_input), axis=-2)  # This should be one value
        return q

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


class DDPG:

    def __init__(self, buffer_size: int = 100, batch_size: int = 64, gamma: float = 0.99,
                 critic_loss: keras.losses.Loss = keras.losses.MeanSquaredError(),
                 critic_optimizer: keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                 actor_loss:  keras.losses.Loss = keras.losses.MeanSquaredError(),
                 actor_optimizer: keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                 tau: float = 0.005):
        self.model = MyModel()
        self.target_model = MyModel()
        self.copied_weights = False

        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, self.batch_size)

        self._gamma = gamma
        self.critic_loss = critic_loss
        self.critic_optimizer = critic_optimizer
        self.actor_loss = actor_loss
        self.actor_optimizer = actor_optimizer
        self._tau = tau

    def _initialize_target(self, state: tf.Tensor) -> None:
        self.target_model.predict(state)
        self._soft_update(self.model.common_weights, self.target_model.common_weights)
        self._soft_update(self.model.critic_weights, self.target_model.critic_weights)
        self._soft_update(self.model.actor_weights, self.target_model.actor_weights)
        self.copied_weights = True

    def train_episode(self) -> float:
        # Run the model for one episode to collect training data
        states, actions, rewards, next_states = run_episode(self.model)

        if not self.copied_weights:
            self._initialize_target(states[1, :, :])

        # Calculate expected returns
        returns = get_expected_return(rewards, self._gamma)

        # Store in buffer
        self.buffer.add(states, actions, returns, next_states)

        # Train per step in buffer (in batch mode)
        s, a, r, ns = [tf.squeeze(b) for b in self.buffer.sample_batch()]

        self._train_critic(s, a, r, ns)
        self._train_actor(s)

        self.update_targets()

        # Return performance for logging
        return float(tf.math.reduce_sum(rewards))

    # @tf.function
    def _train_critic(self, state: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, next_state: tf.Tensor) -> None:
        """
        Use saved (s,a,r,s[t+1]) tuples to build up loss in every step and train only the critic (and common part)
        In forward pass skip actor part since there is no need for it, the critic should use saved actions
        to know what reward to expect
        """

        with tf.GradientTape() as tape:
            # Full forward pass with target networks with the next state s(t+1), this produces q_target
            # Temporal difference (TD) error loss(r + gamma * target_q - q)

            # TD target = r - gamma * target_q
            _, target_q = self.target_model.predict(next_state)
            td_targets = tf.stop_gradient(reward + self._gamma * target_q)

            # Estimate q value
            q = self.model.forward_critic(state, action)

            # TD error loss
            # Could potentially clip error
            critic_loss = self.critic_loss(td_targets, q)

        # Compute the gradients from the loss
        [dl_dcritic, dl_dcommon] = tape.gradient(critic_loss, [self.model.critic_weights, self.model.common_weights])

        # Apply the gradients to the model's parameters
        self.critic_optimizer.apply_gradients(zip(dl_dcritic, self.model.critic_weights))
        self.critic_optimizer.apply_gradients(zip(dl_dcommon, self.model.common_weights))

    # @tf.function
    def _train_actor(self, state: tf.Tensor) -> None:
        """
        Actor gets trained based on critic q values.
        It gets the saved states as input and all the rest is determined from the model, so full forward pass.
        Don't have associated reward for new actions, so can only train actor (and common part).
        """

        with tf.GradientTape() as tape:
            q = self.model.forward_actor(state)

        # Compute the gradients from the loss
        [dl_dactor, dl_dcommon] = tape.gradient(q, [self.model.actor_weights, self.model.common_weights])

        # Apply the gradients to the model's parameters
        self.actor_optimizer.apply_gradients(zip(dl_dactor, self.model.actor_weights))
        self.actor_optimizer.apply_gradients(zip(dl_dcommon, self.model.common_weights))

    def _soft_update(self, source_variables,
                     target_variables,
                     tau=1.0,
                     tau_non_trainable=None):
        op_name = 'soft_variables_update'
        updates = []
        for (v_s, v_t) in zip(source_variables, target_variables):
            if not v_t.trainable:
                current_tau = tau_non_trainable
            else:
                current_tau = tau

            if current_tau == 1.0:
                update = v_t.assign(v_s)
            else:
                update = v_t.assign((1 - current_tau) * v_t + current_tau * v_s)

            updates.append(update)

        return tf.group(*updates, name=op_name)

    def update_targets(self) -> None:
        self._soft_update(self.model.common_weights, self.target_model.common_weights, self._tau)
        self._soft_update(self.model.critic_weights, self.target_model.critic_weights, self._tau)
        self._soft_update(self.model.actor_weights, self.target_model.actor_weights, self._tau)


### Set up training procedure ###

def run_episode(model: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Runs a single episode to collect training data.
    """

    states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    next_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    # initial_state_shape = initial_state.shape
    # state = initial_state

    for t, (state, next_state) in enumerate(wg.train_ds):
        y = next_state[:, :, -2:]
        t = int(t)

        # Run the model and to get action probabilities and critic value
        action, value = model(state)

        # Store state
        states = states.write(t, state)
        next_states = next_states.write(t, next_state)

        # Store chosen action
        actions = actions.write(t, action)

        # Store reward
        rewards = rewards.write(t, float(-tf.math.reduce_sum(tf.math.squared_difference(y, action))))

    states = states.stack()
    actions = actions.stack()
    rewards = rewards.stack()
    next_states = next_states.stack()

    return states, actions, rewards, next_states

def get_expected_return(rewards: tf.Tensor, gamma: float) -> tf.Tensor:
    """
    Compute expected returns per timestep.
    """

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

    return returns


### Set up training step ###
# Learning rate for actor-critic models
# critic_lr = 0.002
# actor_lr = 0.001
# train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
# val_loss = tf.keras.metrics.MeanSquaredError(name='validation_loss')

min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 1000

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

# # Making the weights equal initially
# target_actor.set_weights(actor_model.get_weights())
# target_critic.set_weights(critic_model.get_weights())


# Used to update target networks
# tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
# tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

ddpg = DDPG(buffer_size=100, batch_size=64, gamma=0.99,
            critic_loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            critic_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            actor_loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            actor_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            tau=0.005)

if __name__ == '__main__':

    with trange(max_episodes) as episodes:
        for episode in episodes:

            result = ddpg.train_episode()

            # Log progress
            episodes_reward.append(result)

            # Show average episode reward every 10 episodes
            if episode % 10 == 0:
                print(f'Episode {episode}: last reward: {result}')

    # print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

    # Visualiaze predictions
    # len_test = 20
    #
    predictions1 = []
    targets1 = []
    for i, (x, y) in enumerate(wg.train_ds):
        predictions1.append(ddpg.model(x).numpy()[0, 0, :])
        targets1.append(y.numpy()[0, 0, :])

    targets1 = np.stack(targets1)
    predictions1 = np.stack(predictions1)

    plt.figure(1)
    plt.plot(np.arange(targets1.shape[0]), targets1, predictions1)
    plt.legend(['Target load', 'Target pv',
                'Prediction load', 'Prediction pv'])
    plt.show()
