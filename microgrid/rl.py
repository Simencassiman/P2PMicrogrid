from typing import Union, Optional

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Deque
import collections
import statistics
import random
import gc
from functools import reduce

import config as cf
from config import DB_PATH, TIME_SLOT, MODELS_PATH, CENTS_PER_EURO, HOURS_PER_DAY
import heating
import database
from database import get_connection, get_data

MINUTES_PER_HOUR = 60

"""
Code adapted from
https://github.com/tensorflow/agents/blob/v0.12.0/tf_agents/agents/ddpg/ddpg_agent.py
https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html#References

Other sources
Target networks: https://www.nature.com/articles/nature14236.pdf
DDPG: https://arxiv.org/pdf/1509.02971v2.pdf
Test OU noise: https://rdrr.io/cran/goffda/man/r_ou.html

"""


### Parameter setup ###
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
# env.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


### Create models ###
class QNetwork(keras.Model):
    def __init__(self):
        super(QNetwork, self).__init__()

        self._layers = keras.Sequential([
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(1)
        ])

        self.concat = keras.layers.Concatenate()

    def call(self, state: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        return self._layers(self.concat([state, action]))


class ActorModel:
    def __init__(self, epsilon: float = 0.1):
        self.actions = tf.convert_to_tensor([0, 0.5, 1])
        self._epsilon = epsilon

    def __call__(self, state: tf.Tensor, network: keras.Model, greedy: bool = False) -> tf.Tensor:

        if not greedy and random.random() < self._epsilon:
            # Explore
            action = tf.expand_dims(tf.gather(self.actions, np.random.choice([0, 1, 2], (state.shape[0]))), axis=-1)
        else:
            # Compute Q-values and take best action
            values = tf.TensorArray(dtype=tf.float32, size=self.actions.shape[0])
            for i in range(self.actions.shape[0]):
                values = values.write(i, network(state, tf.repeat(tf.expand_dims(self.actions[i], axis=0)[None, :],
                                                                  state.shape[0], axis=0)))

            action = tf.gather(self.actions, tf.argmax(values.stack(), axis=0))

        return action


class ReplayBuffer:

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.count = 0
        self.buffer: Deque[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]] \
            = collections.deque(maxlen=buffer_size)

    def add(self, s: tf.Tensor, a: tf.Tensor, r: tf.Tensor, ns: tf.Tensor) -> None:
        experience = (s, a, r, ns)

        self.count = min(self.count + 1, self.buffer_size)
        self.buffer.append(experience)

    def add_batch(self, s: tf.Tensor, a: tf.Tensor, r: tf.Tensor, ns: tf.Tensor) -> None:
        for i in tf.range(tf.shape(s)[0]):
            experience = (s[i, :], a[i], r[i], ns[i, :])

            self.count = min(self.count + 1, self.buffer_size)
            self.buffer.append(experience)

    def size(self) -> int:
        return min(self.count, self.buffer_size)

    def sample_batch(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        """

        if self.count < self.batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, self.batch_size)

        s_batch = tf.stack([_[0] for _ in batch])
        a_batch = tf.stack([_[1] for _ in batch])
        r_batch = tf.stack([_[2] for _ in batch])
        ns_batch = tf.stack([_[3] for _ in batch])

        return s_batch, a_batch, r_batch, ns_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class Trainer:

    def __init__(self, buffer_size: int, batch_size: int,
                 gamma: float, epsilon: float, tau: float,
                 optimizer: tf.optimizers.Optimizer):
        self._gamma = gamma
        self._tau = tau

        self.actor = ActorModel(epsilon)

        self.q_network = QNetwork()
        self.target_q_network = QNetwork()

        self.optimizer = optimizer

        self.buffer = ReplayBuffer(buffer_size, batch_size)

        self._initialize_buffer()
        self._initialize_target()

    def _initialize_buffer(self) -> None:
        while self.buffer.size() < 100:
            self.buffer.add_batch(*run_episode(self.actor, self.q_network))

    def _initialize_target(self) -> None:
        s, a, _, _ = self.buffer.sample_batch()
        _ = self.target_q_network(s, a)
        self._soft_update(self.q_network.trainable_weights, self.target_q_network.trainable_weights)

    def train_episode(self) -> float:

        states, actions, rewards, next_states = run_episode(self.actor, self.q_network)

        for i in range(states.shape[0]):

            self.buffer.add(states[i, :], actions[i], rewards[i], next_states[i, :])

            s, a, r, ns = self.buffer.sample_batch()

            self._train(s, a, r, ns)
            self.update_targets()

        return float(tf.math.reduce_sum(rewards).numpy())

    @tf.function
    def _train(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor, next_states: tf.Tensor) -> None:

        with tf.GradientTape() as tape:
            q_target = rewards + self._gamma * self.actor(next_states, self.target_q_network, greedy=True)
            q_value = self.q_network(states, actions)

            loss = tf.math.reduce_mean(tf.math.squared_difference(q_target, q_value))

        dl_dw = tape.gradient(loss, self.q_network.trainable_weights)

        self.optimizer.apply_gradients(zip(dl_dw, self.q_network.trainable_weights))

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
        self._soft_update(self.q_network.trainable_weights, self.target_q_network.trainable_weights, self._tau)


### Prepare data ###

### Set up training procedure ###

def run_episode(model: ActorModel,
                q_network: QNetwork) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Runs a single episode to collect training data.
    """

    states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    next_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    # initial_state_shape = initial_state.shape
    # state = initial_state
    t_in = 21.0 + np.random.normal()
    t_bm = 21.0 + np.random.normal()
    cop = 3.
    max_power = 3e3

    for t, ((state, next_state), p) in enumerate(zip(ds, price)):
        t = int(t)

        # Extract outside temperature from state and put indoor temperature in
        state = state.numpy()
        t_out, state[1] = state[1], t_in
        state = tf.convert_to_tensor(state)

        # Run the model and to get action probabilities and critic value
        action = model(tf.expand_dims(state, axis=0), q_network)
        scaled_action = action * max_power

        # Compute temperature evolution
        t_in, t_bm = heating.temperature_simulation(t_out, t_in, t_bm, scaled_action, cop)

        # Store state
        states = states.write(t, state)
        next_state = next_state.numpy()
        next_state[1] = t_in
        next_states = next_states.write(t, tf.convert_to_tensor(next_state))

        # Store chosen action
        actions = actions.write(t, action)

        # Store reward
        p_out = (state[2] + scaled_action) / 1e3
        cost = tf.where(p_out >= 0, p_out * p, p_out * 0.07)
        t_penalty = max(max(0., 20. - t_in), max(0., t_in - 22.))
        t_penalty = tf.where(t_penalty > 0, t_penalty + 1, 0)
        r = - (cost + 10 * t_penalty ** 2)
        rewards = rewards.write(t, r)

    states = states.stack()
    actions = actions.concat()
    rewards = rewards.concat()
    next_states = next_states.stack()

    return states, actions, rewards, next_states


def run_single_trial(trainer: Trainer) -> float:
    with trange(starting_episodes, max_episodes) as episodes:
        for episode in episodes:

            result = trainer.train_episode()

            # Log progress
            episodes_reward.append(result)

            # Show average episode reward every x episodes
            if episode % min_episodes_criterion == 0:
                # Compute statistics
                training = statistics.mean(episodes_reward)

                # Report results
                print(f'Episode {episode}: running reward: {training:.3f}')

    return training


def test(model: ActorModel, q_network: QNetwork) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Runs a single episode to test performance.
    """

    states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    temperatures = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    costs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    # initial_state_shape = initial_state.shape
    # state = initial_state
    t_in = 21.0
    t_bm = 21.0
    cop = 3.
    max_power = 3e3

    for t, ((state, _), p) in enumerate(zip(ds, price)):
        t = int(t)

        # Extract outside temperature from state and put indoor temperature in
        state = state.numpy()
        t_out, state[1] = state[1], t_in
        state = tf.convert_to_tensor(state)

        # Run the model and to get action probabilities and critic value
        action = model(tf.expand_dims(state, axis=0), q_network)
        scaled_action = action * max_power

        # Compute temperature evolution
        t_in, t_bm = heating.temperature_simulation(t_out, t_in, t_bm, scaled_action, cop)

        # Store state
        states = states.write(t, state)
        temperatures = temperatures.write(t, t_in)

        # Store chosen action
        actions = actions.write(t, scaled_action)

        # Calculate cost
        p_out = (state[2] * balance_max + scaled_action) / 1e3
        cost = tf.where(p_out >= 0, p_out * p, p_out * 0.07)
        costs = costs.write(t, -cost)

    states = states.stack()
    actions = actions.concat()
    temperatures = temperatures.concat()
    costs = costs.concat()

    return states, actions, temperatures, costs


### Set up training ###
trials = 3
min_episodes_criterion = 100
starting_episodes = 0 * 1000
max_episodes = 500
# max_steps_per_episode = 1000

mse_loss = tf.keras.losses.MeanSquaredError()

bu = 100 * 1000
bs = 128
lr = 1e-5
gamma = 0.95
tau = 0.005
epsilon = 0.1

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)


if __name__ == '__main__':

    data = np.array(train_df_1, dtype=np.float32)
    ds = train_dataset = tf.data.Dataset.from_tensor_slices((data, np.roll(data, -1, axis=0)))

    dqn = Trainer(bu, bs, gamma, epsilon, tau,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    run_single_trial(dqn)

    s, a, temps, cost = test(dqn.actor, dqn.q_network)

    print(f'Price paid: {cost.numpy().sum()}')

    p_out = (s[:, -1] * balance_max + a)[:, 0] / 1e3

    fig_1 = plt.figure(1)
    plt.plot(np.arange(s.shape[0]),
             s[:, -1] * balance_max / 1e3,
             a[:, 0] / 1e3)
    plt.plot(np.arange(s.shape[0]), (s[:, -1] * balance_max + a[:, 0]) / 1e3)

    fig_2, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(np.arange(s.shape[0]), temps)
    ax2.plot(np.arange(s.shape[0]), price, 'g-')

    ax1.set_ylabel('Temperature [°C]')
    ax2.set_ylabel('Price [€]', color='g')

    fig_3 = plt.figure(3)
    plt.plot(np.arange(s.shape[0]), cost)

    plt.show()

    # for b, bs in enumerate([128]):
    #
    #     # Set up settings string for saving the results
    #     hyperparams = f'activation={activation};episodes={max_episodes};bu={bu};bs={bs};gamma={gamma};ls={lr};'\
    #                   f'tau={tau};sd={sd};theta={theta};sigma={sigma};horizon={horizon};clr=x2'
    #
    #     print(f'----- {hyperparams} ----- ')
    #
    #     # Run trials to average results
    #     model_acc = np.zeros(trials)
    #     for trial in range(trials):
    #         # Reinitialize
    #         ddpg = DDPG(outputs=nr_outputs, buffer_size=bu, batch_size=bs, gamma=gamma,
    #                     critic_loss=mse_loss,
    #                     actor_loss=mse_loss,
    #                     critic_optimizer=tf.keras.optimizers.Adam(learning_rate=lr * 2),
    #                     actor_optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    #                     tau=tau, theta=theta, sigma=sigma, sd=sd)
    #         episodes_reward.clear()
    #
    #         # Load partially trained models_ddpg
    #         # prev_hyperparams = f'activation=sigmoid;episodes={starting_episodes};bu={bu};bs={bs};' \
    #         #                    f'gamma={gamma};ls={lr};tau={tau};sd={sd};theta={theta};sigma={sigma};' \
    #         #                    f'horizon={horizon}'
    #         # load_models('checkpoints', prev_hyperparams, trial, ddpg.actor, ddpg.critic,
    #         #             ddpg.target_actor, ddpg.target_critic)
    #
    #         print(f'----- Running trial {trial + 1} -----')
    #
    #         # Train
    #         result = run_single_trial(ddpg, hyperparams, trial)
    #
    #         # Log results
    #         model_acc[trial] = result
    #
    #     # Log bets performer to test later
    #     best_trial = int(np.argmax(model_acc))
    #     ddpg = DDPG(outputs=nr_outputs, buffer_size=bu, batch_size=bs, gamma=gamma,
    #                 critic_loss=mse_loss,
    #                 actor_loss=mse_loss,
    #                 critic_optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    #                 actor_optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    #                 tau=tau, theta=theta, sigma=sigma)
    #
    #     # Save best checkpointed model
    #     load_models('checkpoints', hyperparams, best_trial, ddpg.actor,
    #                 ddpg.critic, ddpg.target_actor, ddpg.target_critic)
    #     # Save it as best trial
    #     save_models('best_trials', hyperparams, best_trial, ddpg.actor,
    #                 ddpg.critic, ddpg.target_actor, ddpg.target_critic)
    #
    #     # Store predictions
    #     targets, predictions, errors, q_estimates = check_performance(ddpg, hyperparams)
    #
    #     # Plot predictions
    #     plt.figure(10 * b)
    #     plt.plot(np.arange(targets.shape[0]), targets[:, -1:], predictions)
    #     plt.title(hyperparams)
    #     plt.legend(['Target load',
    #                 'Prediction load'])
    #
    #     plt.figure(10 * b + 1)
    #     plt.plot(np.arange(targets.shape[0]), errors, q_estimates)
    #     plt.title(hyperparams)
    #     plt.legend(['Error',
    #                 'Q'])
    #
    #     # Garbage collection
    #     del ddpg
    #     gc.collect()
    #
    # plt.show()
