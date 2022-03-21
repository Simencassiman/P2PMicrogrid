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

import config
from config import DB_PATH, TIME_SLOT, MODELS_PATH
import database
from database import get_connection, get_data
import ml
from ml import WindowGenerator

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
    def __init__(self, outputs: int):
        super(ActorModel, self).__init__()

        self.lstm = keras.layers.LSTM(100, return_sequences=True)

        self.actor = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(100, activation='relu'),
            self.lstm,
            self.lstm,
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(outputs, activation='sigmoid')
        ])

    def call(self, state: tf.Tensor) -> tf.Tensor:
        return self.actor(state)


class CriticModel(keras.Model):
    def __init__(self, outputs):
        super(CriticModel, self).__init__()

        self.lstm = keras.layers.LSTM(100, return_sequences=True)
        self.concat = keras.layers.Concatenate()

        self.critic = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(100, activation='relu'),
            self.lstm,
            self.lstm,
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(outputs)
        ])

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        q_vals = self.critic(self.concat(inputs))
        return tf.math.reduce_sum(q_vals, axis=-2)


class ReplayBuffer:

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.count = 0
        self.buffer: Deque[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]] \
            = collections.deque(maxlen=buffer_size)

    def add(self, s: tf.Tensor, a: tf.Tensor, r: tf.Tensor, ns: tf.Tensor) -> None:
        for i in tf.range(tf.shape(s)[0]):
            experience = (s[i, :, :], a[i, :, :], r[i], ns[i, :, :])

            self.count += 1
            self.buffer.append(experience)

    def size(self) -> int:
        return min(self.count, self.buffer_size)

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray,np.ndarray, np.ndarray]:
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

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        ns_batch = np.array([_[3] for _ in batch])

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

    def __init__(self, outputs: int, buffer_size: int = 1000, batch_size: int = 64, gamma: float = 0.99,
                 critic_loss: keras.losses.Loss = keras.losses.MeanSquaredError(),
                 actor_loss:  keras.losses.Loss = keras.losses.MeanSquaredError(),
                 critic_optimizer: keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
                 actor_optimizer: keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
                 tau: float = 0.005, theta: float = 0.1, sigma: float = 0.1, sd: float = 0.2):

        self.actor = ActorModel(outputs)
        self.target_actor = ActorModel(outputs)

        self.critic = CriticModel(outputs=1)
        self.target_critic = CriticModel(outputs=1)

        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros((1, 3, 1)), theta=theta, sigma=sigma, sd=sd)

        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, self.batch_size)

        self._gamma = gamma
        self.critic_loss = critic_loss
        self.critic_optimizer = critic_optimizer
        self.actor_loss = actor_loss
        self.actor_optimizer = actor_optimizer
        self._tau = tau

        self._initialize_target()

    def _initialize_target(self) -> None:
        # target_actor.set_weights(actor_model.get_weights())
        # target_critic.set_weights(critic_model.get_weights())
        state, action, _, _ = run_episode(self.actor, self.actor_noise)
        self.critic([state, action])
        state, action, _, _ = run_episode(self.target_actor, self.actor_noise)
        self.target_critic([state, action])

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

    # @tf.function
    def train_episode(self) -> float:
        # Run the model for one episode to collect training data
        states, actions, rewards, next_states = run_episode(self.actor, self.actor_noise)

        # Calculate expected returns
        returns = self.get_expected_return(rewards, tf.expand_dims(next_states[-1, :, :], axis=0))

        # Store in buffer
        self.buffer.add(states, actions, returns, next_states)

        # Train per step in buffer (in batch mode)
        s, a, r, ns = self.buffer.sample_batch()
        self._train_critic(s, a, r, ns)
        self._train_actor(s)

        self.update_targets()

        # Return performance for logging
        return float(tf.math.reduce_sum(rewards))

    @tf.function
    def _train_critic(self, state: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, next_state: tf.Tensor) -> None:
        # Use saved (s,a,r,s[t+1]) tuples to build up loss in every step and train only the critic (and common part)
        # In forward pass skip actor part since there is no need for it, the critic should use saved actions
        # to know what reward to expect

        with tf.GradientTape() as tape:
            # Estimate q value
            q = self.critic([state, action])

            critic_loss = self.critic_loss(tf.squeeze(reward), q)

        # Compute the gradients from the loss
        dl_dcritic = tape.gradient(critic_loss, self.critic.trainable_weights)
        dl_dcritic[0] = tf.clip_by_value(dl_dcritic[0], -1., 1.)

        # Apply the gradients to the model's parameters
        self.critic_optimizer.apply_gradients(zip(dl_dcritic, self.critic.trainable_weights))

    @tf.function
    def _train_actor(self, state: tf.Tensor) -> None:
        # Actor gets trained based on critic q values.
        # It gets the saved states as input and all the rest is determined from the model, so full forward pass.
        # Don't have associated reward for new actions, so can only train actor (and common part).

        with tf.GradientTape() as tape:
            action = self.actor(state)
            q = -self.critic([state, action])   # Negative of Q value because we want to improve the action

        # Compute the gradients from the loss
        dl_dactor = tape.gradient(q, self.actor.trainable_weights)
        dl_dactor[0] = tf.clip_by_value(dl_dactor[0], -1., 1.)

        # Apply the gradients to the model's parameters
        self.actor_optimizer.apply_gradients(zip(dl_dactor, self.actor.trainable_weights))

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
        self._soft_update(self.critic.trainable_weights, self.target_critic.trainable_weights, self._tau)
        self._soft_update(self.actor.trainable_weights, self.target_actor.trainable_weights, self._tau)


### Prepare data ###

def compute_time_slot(time) -> int:
    t = datetime.strptime(time, '%H:%M:%S')

    return (t.minute / TIME_SLOT) + t.hour * MINUTES_PER_HOUR / TIME_SLOT


conn = get_connection(DB_PATH)
start = datetime(2021, 11, 1)
val_start = datetime(2021, 11, 2)
val_end = datetime(2021, 11, 2)
end = datetime(2021, 11, 2)
df = get_data(conn, start, end)

df['time'] = df['time'].map(lambda t: compute_time_slot(t))
df[['year', 'month', 'day']] = df['date'].str.split(pat='-', expand=True)
df['time'] = df['time'] / 96.
df['day'] = df['day'].astype(float) / 31.
df['month'] = df['month'].astype(float) / 12.
df['l0'] = df['l0'].astype(float) / (df['l0'].max().astype(float) * 1.1)
df['pv'] = df['pv'].astype(float) / (df['pv'].max().astype(float) * 1.1)
df['date'] = df['date'].map(lambda t: datetime.strptime(t, '%Y-%m-%d'))

nr_outputs = 1
outputs = ['l0', 'pv']
cols = ['time', 'day', 'month'] + outputs[:nr_outputs]
# cols = ['time', 'day', 'month', 'temperature', 'cloud_cover', 'humidity'] + outputs[:nr_outputs]
train_df_1 = df[df['date'] < val_start][cols]
train_df_2 = df[df['date'] > val_end][cols]
val_df = df[(val_start <= df['date']) & (df['date'] <= val_end)][cols]

horizon = 3
wg = WindowGenerator(train_df=train_df_1, val_df=None,
                     input_width=horizon, label_width=horizon, label_columns=list(val_df.columns),
                     batch_size=96)


### Set up training procedure ###

def run_episode(model: tf.keras.Model,
                actor_noise: OrnsteinUhlenbeckActionNoise) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Runs a single episode to collect training data.
    """

    states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    next_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    # initial_state_shape = initial_state.shape
    # state = initial_state
    actor_noise.reset()

    for t, (state, next_state) in enumerate(wg.train_ds):
        y = next_state[:, :, -nr_outputs:]
        t = int(t)

        # Run the model and to get action probabilities and critic value
        action = model(state)
        action = action + actor_noise()

        # Store state
        states = states.write(t, state)
        next_states = next_states.write(t, next_state)

        # Store chosen action
        actions = actions.write(t, action)

        # Store reward
        rewards = rewards.write(t, -tf.math.reduce_sum(tf.math.squared_difference(y, action), axis=-2))

    states = states.concat()
    actions = actions.concat()
    rewards = rewards.concat()
    next_states = next_states.concat()

    return states, actions, rewards, next_states


def test(model: tf.keras.Model) -> tf.Tensor:
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    for t, (state, next_state) in enumerate(wg.train_ds):
        # Run the model and to get action probabilities and critic value
        action = model(state)

        # Store reward
        rewards = rewards.write(int(t),
                                -tf.math.reduce_sum(tf.math.squared_difference(next_state[:, :, -nr_outputs:],
                                                                               action), axis=-2))
    return rewards.concat()


def get_model_path(folder: str, settings: str, trial: int, model: str) -> str:
    return f'{MODELS_PATH}/{folder}/{settings}_{trial}/{model}'


def save_models(folder: str, settings: str, trial: int, actor: keras.Model,
                critic: keras.Model, target_actor: keras.Model, target_critic: keras.Model) -> None:
    """
    Saves model weight so they can be retrieved from a file.

    :param folder: folder inside model where to store the weights
    :param settings: hyperparameter settings string
    :param trial: integer indicating the run
    """

    actor.save_weights(get_model_path(folder, settings, trial, 'actor'))
    critic.save_weights(get_model_path(folder, settings, trial, 'critic'))
    target_actor.save_weights(get_model_path(folder, settings, trial, 'target_actor'))
    target_critic.save_weights(get_model_path(folder, settings, trial, 'target_critic'))


def load_models(folder: str, settings: str, trial: int, actor: keras.Model,
                critic: keras.Model, target_actor: keras.Model, target_critic: keras.Model) -> None:

    actor.load_weights(get_model_path(folder, settings, trial, 'actor'))
    critic.load_weights(get_model_path(folder, settings, trial, 'critic'))
    target_actor.load_weights(get_model_path(folder, settings, trial, 'target_actor'))
    target_critic.load_weights(get_model_path(folder, settings, trial, 'target_critic'))


def run_single_trial(trainer: DDPG, settings: str, trial: int) -> float:
    with trange(starting_episodes, max_episodes) as episodes:
        for episode in episodes:

            result = trainer.train_episode()

            # Log progress
            episodes_reward.append(result)

            # Show average episode reward every x episodes
            if episode % min_episodes_criterion == 0:
                # Compute statistics
                training = statistics.mean(episodes_reward)
                validation = float(tf.math.reduce_sum(test(trainer.actor)).numpy())

                # Save results
                database.log_training(conn, settings, trial, episode, training, validation)

                # Save weights
                save_models('checkpoints', settings, trial,
                            trainer.actor, trainer.critic, trainer.target_actor, trainer.target_critic)

                # Report results
                print(f'Episode {episode}: running reward: {training:.3f}, validation: {validation:.3f}')

    return validation


def check_performance(trainer: DDPG, settings: str) -> Tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []

    for x, y in wg.train_ds:
        actions = trainer.actor(x)

        for target, prediction in tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(y),
                                                       tf.data.Dataset.from_tensor_slices(actions))):
            predictions.append(prediction[-1, :].numpy())
            targets.append(target[-1, :].numpy())

    targets = np.stack(targets)
    predictions = np.stack(predictions)

    # Store in db
    zeros = np.zeros(predictions.shape)[:, 0].tolist()
    database.log_predictions(conn, settings,
                             (targets[:, 1] + 100 * targets[:, 2]).tolist(), targets[:, 0],
                             predictions[:, 0].tolist(), zeros,
                             targets[:, -1].tolist(), zeros)

    return targets, predictions


### Set up training ###
trials = 3
min_episodes_criterion = 100
starting_episodes = 5 * 1000
max_episodes = 10 * 1000
# max_steps_per_episode = 1000

mse_loss = tf.keras.losses.MeanSquaredError()

bu = 10000
bs = 32
gamma = 0.95
lr = 1e-7
tau = 0.005
theta = 0.1
sigma = 0.1
sd = 0.2
activation = 'sigmoid'

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)


if __name__ == '__main__':

    for g, gamma in enumerate([0.90, 0.95, 0.99]):
        for l, lr in enumerate([1e-6, 1e-7]):
            for b, bs in enumerate([32, 64, 128]):
                # Set up settings string for saving the results
                hyperparams = f'activation={activation};episodes={max_episodes};bu={bu};bs={bs};gamma={gamma};ls={lr};'\
                              f'tau={tau};sd={sd};theta={theta};sigma={sigma};horizon={horizon}'

                print(f'----- {hyperparams} ----- ')

                # Run trials to average results
                model_acc = np.zeros(trials)
                for trial in range(trials):
                    # Reinitialize
                    ddpg = DDPG(outputs=nr_outputs, buffer_size=bu, batch_size=bs, gamma=gamma,
                                critic_loss=mse_loss,
                                actor_loss=mse_loss,
                                critic_optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                actor_optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                tau=tau, theta=theta, sigma=sigma, sd=sd)
                    episodes_reward.clear()

                    # Load partially trained models
                    prev_hyperparams = f'activation=sigmoid;episodes={starting_episodes};bu={bu};bs={bs};' \
                                       f'gamma={gamma};ls={lr};tau={tau};sd={sd};theta={theta};sigma={sigma};' \
                                       f'horizon={horizon}'
                    load_models('checkpoints', prev_hyperparams, trial, ddpg.actor, ddpg.critic,
                                ddpg.target_actor, ddpg.target_critic)

                    print(f'----- Running trial {trial + 1} -----')

                    # Train
                    result = run_single_trial(ddpg, hyperparams, trial)

                    # Log results
                    model_acc[trial] = result

                # Log bets performer to test later
                best_trial = int(np.argmax(model_acc))
                ddpg = DDPG(outputs=nr_outputs, buffer_size=bu, batch_size=bs, gamma=gamma,
                            critic_loss=mse_loss,
                            actor_loss=mse_loss,
                            critic_optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                            actor_optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                            tau=tau, theta=theta, sigma=sigma)

                # Save best checkpointed model
                load_models('checkpoints', hyperparams, best_trial, ddpg.actor,
                            ddpg.critic, ddpg.target_actor, ddpg.target_critic)
                # Save it as best trial
                save_models('best_trials', hyperparams, best_trial, ddpg.actor,
                            ddpg.critic, ddpg.target_actor, ddpg.target_critic)

                # Store predictions
                targets, predictions = check_performance(ddpg, hyperparams)

                # Plot predictions
                plt.figure(100*g + 10 * l + b)
                plt.plot(np.arange(targets.shape[0]), targets[:, -1:], predictions)
                plt.title(hyperparams)
                plt.legend(['Target load',
                            'Prediction load'])

                # Garbage collection
                gc.collect()

    plt.show()
