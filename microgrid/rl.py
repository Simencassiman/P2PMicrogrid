import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, List, Deque
import collections
import statistics
import random
import gc

import config as cf
from config import DB_PATH, TIME_SLOT, MODELS_PATH, CENTS_PER_EURO, HOURS_PER_DAY
import database
from database import get_connection, get_data
import ml
from ml import WindowGenerator
import heating
import dataset as ds

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


### Create model ###
class ActorModel(keras.Model):
    def __init__(self) -> None:
        super(ActorModel, self).__init__()

        self._layers = keras.Sequential([
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(1)
        ])

        self.concat = keras.layers.Concatenate()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._layers(x)


class CriticModel(keras.Model):
    def __init__(self) -> None:
        super(CriticModel, self).__init__()

        self._layers = keras.Sequential([
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(1)
        ])

        self.concat = keras.layers.Concatenate()

    def call(self, state: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        return self._layers(self.concat([state, action]))


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


class DDPG:

    def __init__(self, buffer_size: int = 1000, batch_size: int = 32, gamma: float = 0.99,
                 critic_loss: keras.losses.Loss = keras.losses.MeanSquaredError(),
                 actor_loss:  keras.losses.Loss = keras.losses.MeanSquaredError(),
                 critic_optimizer: keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
                 actor_optimizer: keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
                 tau: float = 0.005, theta: float = 0.1, sigma: float = 0.1, sd: float = 0.2):

        self.actor = ActorModel()
        self.target_actor = ActorModel()

        self.critic = CriticModel()
        self.target_critic = CriticModel()

        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, self.batch_size)

        self._gamma = gamma
        self.critic_loss = critic_loss
        self.critic_optimizer = critic_optimizer
        self.actor_loss = actor_loss
        self.actor_optimizer = actor_optimizer
        self._tau = tau

        self._initialize_buffer()
        self._initialize_target()

    def _initialize_buffer(self) -> None:
        while self.buffer.count < 2000:
            states, actions, rewards, next_states = run_episode(self.actor)
            self.buffer.add_batch(states, actions, rewards, next_states)

    def _initialize_target(self) -> None:
        s, a, _, _ = self.buffer.sample_batch()
        _ = self.target_actor(a)
        _ = self.target_critic([s, a])

        self._soft_update(self.actor.trainable_weights, self.target_actor.trainable_weights)
        self._soft_update(self.critic.trainable_weights, self.target_critic.trainable_weights)

    def predict(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        action = self.actor(state)
        q_value = self.critic([state, action])

        return action, q_value

    def predict_target(self, state: tf.Tensor) -> tf.Tensor:
        action = self.target_actor(state)
        q_value = self.target_critic([state, action])

        return q_value

    def get_expected_return(self, rewards: tf.Tensor, next_state: tf.Tensor) -> tf.Tensor:

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = self.predict_target(next_state)  # tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self._gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)

        returns = returns.stack()[::-1]

        return returns

    def train_episode(self) -> float:
        # Run the model for one episode to collect training data
        states, actions, rewards, next_states = run_episode(self.actor)

        # Calculate expected returns
        # returns = self.get_expected_return(rewards, tf.expand_dims(next_states[-1, :, :], axis=0))

        # Store in buffer
        # self.buffer.add(states, actions, returns, next_states)
        for s, a, r, ns in tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(states),
                                                tf.data.Dataset.from_tensor_slices(actions),
                                                tf.data.Dataset.from_tensor_slices(rewards),
                                                tf.data.Dataset.from_tensor_slices(next_states),)):
            self.buffer.add(s, a, r, ns)

            # Train per step in buffer (in batch mode)
            sb, ab, rb, nsb = self.buffer.sample_batch()
            self.update(sb, ab, rb, nsb)
            self.update_targets()

        # Return performance for logging
        return float(tf.math.reduce_sum(rewards))

    @tf.function
    def update(self, state: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, next_state: tf.Tensor) -> None:
        # Use saved (s,a,r,s[t+1]) tuples to build up loss in every step and train only the critic (and common part)
        # In forward pass skip actor part since there is no need for it, the critic should use saved actions
        # to know what reward to expect

        # ---- Train Critic ---- #
        with tf.GradientTape() as tape:
            # Estimate q value
            q = self.critic([state, action])

            # TD error
            y = reward + self._gamma * self.predict_target(next_state)
            critic_loss = tf.math.reduce_mean(tf.math.squared_difference(y, q))

        # Compute the gradients from the loss
        dl_dcritic = tape.gradient(critic_loss, self.critic.trainable_weights)
        dl_dcritic[0] = tf.clip_by_value(dl_dcritic[0], -1., 1.)

        # Apply the gradients to the model's parameters
        self.critic_optimizer.apply_gradients(zip(dl_dcritic, self.critic.trainable_weights))

        # ---- Train actor ---- #
        # Actor gets trained based on critic q values.
        # It gets the saved states as input and all the rest is determined from the model, so full forward pass.
        # Don't have associated reward for new actions, so can only train actor (and common part).
        with tf.GradientTape() as tape:
            action = self.actor(state)
            q = -tf.math.reduce_mean(self.critic([state, action]))      # Negative of Q-value because
                                                                        # we want to improve the action

        # Compute the gradients from the loss
        dq_dactor = tape.gradient(q, self.actor.trainable_weights)
        dq_dactor[0] = tf.clip_by_value(dq_dactor[0], -1., 1.)

        # Apply the gradients to the model's parameters
        self.actor_optimizer.apply_gradients(zip(dq_dactor, self.actor.trainable_weights))

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
        self._soft_update(self.actor.trainable_weights, self.target_actor.trainable_weights, self._tau)
        self._soft_update(self.critic.trainable_weights, self.target_critic.trainable_weights, self._tau)


# ------- Prepare data ------- #

# Get data from database
env_df, agent_df = ds.get_train_data()

# Calculate agent's electricity balance between load and pv generation
env_df['balance'] = agent_df['l0'] * 0.7e3 - agent_df['pv'] * 4e3
balance_max = env_df['balance'].max()
env_df['balance'] = env_df['balance'] / balance_max

# Generate prices
env_df['price'] = (
            (cf.GRID_COST_AVG
             + cf.GRID_COST_AMPLITUDE
             * np.sin(2 * np.pi * np.array(env_df['time']) * cf.HOURS_PER_DAY / cf.GRID_COST_PERIOD
                      + cf.GRID_COST_PHASE)
             ) / cf.CENTS_PER_EURO  # from c€ to €
    )

# Transform to dataset
data = ds.dataframe_to_dataset(env_df)


# ------- Set up training procedure ------- #

def run_episode(model: ActorModel) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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

    for t, (state, next_state) in enumerate(data):
        p = state[-1]
        t = int(t)

        # Extract outside temperature from state and put indoor temperature in
        state = state.numpy()
        t_out, state[1] = state[1], t_in
        state = tf.convert_to_tensor(state)

        # Run the model and to get action probabilities and critic value
        action = model(tf.expand_dims(state, axis=0))
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


def run_single_trial(trainer: DDPG) -> float:
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


def test(model: ActorModel) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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

    for t, (state, _) in enumerate(data):
        p = state[-1]
        t = int(t)

        # Extract outside temperature from state and put indoor temperature in
        state = state.numpy()
        t_out, state[1] = state[1], t_in
        state = tf.convert_to_tensor(state)

        # Run the model and to get action probabilities and critic value
        action = model(tf.expand_dims(state, axis=0))
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


# ------- Set up training ------- #
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

    env_df, _ = ds.get_train_data()
    env_ds = ds.dataframe_to_dataset(env_df)

    price = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for t, (state, _) in enumerate(env_ds):
        price = price.write(t,
                (cf.GRID_COST_AVG
                 + cf.GRID_COST_AMPLITUDE
                 * tf.math.sin(state[0] * 2 * np.pi * cf.HOURS_PER_DAY / cf.GRID_COST_PERIOD - cf.GRID_COST_PHASE)
                 ) / cf.CENTS_PER_EURO  # from c€ to €
        )

    price = price.stack()
    print(price.shape)
    # dqn = Trainer(ActorModel(epsilon),
    #               bu, bs, gamma, tau,
    #               optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    #
    # run_single_trial(dqn)
    #
    # s, a, temps, cost = test(dqn.actor)
    #
    # p_out = (s[:, -1] * balance_max + a)[:, 0] / 1e3
    #
    # print(f'Price paid: {cost.numpy().sum() - 50 / 365 * max(2.5, p_out.numpy().max())}')
    #
    # fig_1 = plt.figure(1)
    # plt.plot(np.arange(s.shape[0]),
    #          s[:, -2] * balance_max / 1e3)
    # plt.plot(np.arange(s.shape[0]), a[:, 0] / 1e3)
    #
    # fig_2, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(np.arange(s.shape[0]), temps)
    # ax2.plot(np.arange(s.shape[0]), env_df['price'], 'g-')
    #
    # ax1.set_ylabel('Temperature [°C]')
    # ax2.set_ylabel('Price [€]', color='g')
    #
    plt.plot(env_df['time'] * 24, price)
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
