# Python Libraries
from __future__ import annotations

import collections
import json
import sqlite3
import statistics
import time
import traceback
from typing import List, Tuple, Callable, Any, Optional
import numpy as np

import pandas as pd
import tensorflow as tf
from tqdm import trange

# Local Modules
from environment import env
from config import TIME_SLOT, MINUTES_PER_HOUR, HOURS_PER_DAY
from agent import Agent, ActingAgent, GridAgent, RuleAgent, RLAgent, QAgent, BaselineAgent
from production import Prosumer, PV
from storage import NoStorage
from heating import HPHeating, HeatPump
from data_analysis import analyse_community_output
import dataset as ds
import database as db


# ------- Parameter setup ------- #
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def get_community(agent_constructor: Callable[[Any], ActingAgent], n_agents: int) -> CommunityMicrogrid:
    # Load time series
    env_df, agent_df = ds.get_train_data()

    timeline = env_df['time'].map(lambda t: int(t * MINUTES_PER_HOUR / TIME_SLOT * HOURS_PER_DAY))
    # timeline = np.array([datetime.fromisoformat(date) for date in data['isodate']])

    agents: List[ActingAgent] = []

    load_ratings = np.array([0.7] * n_agents)  # np.random.normal(0.7, 0.2, n_agents)
    pv_ratings = np.array([4] * n_agents)  # np.random.normal(4, 0.2, n_agents)

    # Create agents
    Agent.reset_ids()
    for i in range(n_agents):
        max_power = max(load_ratings[i], pv_ratings[i])
        safety = 1.1

        agent_load = ds.dataframe_to_dataset(agent_df['l0'] * load_ratings[i] * 1e3)
        agent_pv = ds.dataframe_to_dataset(agent_df['pv'] * pv_ratings[i] * 1e3)

        agents.append(agent_constructor(agent_load,
                                        Prosumer(PV(peak_power=pv_ratings[i] * 1e3,
                                                    production=agent_pv)),
                                        NoStorage(),
                                        HPHeating(HeatPump(cop=3.0, max_power=3 * 1e3, power=tf.constant([0.0])), 21.0),
                                        max_in=max_power * safety * 1e3,
                                        max_out=-(max_power + safety * 1e3)
        ))

    # Prepare environment
    env.setup(ds.dataframe_to_dataset(env_df))

    comm = CommunityMicrogrid(timeline, agents) if n_agents > 1 else SingleAgentCommunity(timeline, agents)

    return comm


def get_rule_based_community(n_agents: int = 5) -> CommunityMicrogrid:
    return get_community(RuleAgent, n_agents)


def get_rl_based_community(n_agents: int = 5) -> CommunityMicrogrid:
    if implementation == 'tabular':
        return get_community(QAgent, n_agents)
    if implementation == 'dqn':
        return get_community(RLAgent, n_agents)

def get_baseline_community() -> SingleAgentCommunity:
    return get_community(BaselineAgent, 1)


class CommunityMicrogrid:

    def __init__(self, timeline: pd.DataFrame, agents: List[ActingAgent]) -> None:
        self.timeline = timeline
        self.time_length = len(timeline)
        self.agents = agents
        self.grid = GridAgent()
        self.q = np.zeros((len(env), len(agents), 3))

        self._rounds = 1

    def _assign_powers(self, p2p_power: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        p_match = tf.where(tf.math.sign(p2p_power) != tf.math.sign(tf.transpose(p2p_power)),
                           p2p_power + tf.transpose(p2p_power),
                           0.)
        exchange = tf.math.sign(p_match) * tf.math.minimum(tf.math.abs(p_match), tf.transpose(tf.math.abs(p_match)))

        p_grid = tf.math.reduce_sum(p2p_power - exchange, axis=1)
        p_p2p = tf.math.reduce_sum(exchange, axis=1)

        return p_grid, p_p2p

    def _compute_costs(self, grid_power: tf.Tensor, peer_power: tf.Tensor,
                       buying_price: tf.Tensor, injection_price: tf.Tensor, p2p_price: tf.Tensor) -> tf.Tensor:
        costs = (
            tf.where(grid_power >= 0.,
                     grid_power * buying_price[:, None],
                     grid_power * injection_price[:, None])
            + peer_power * p2p_price[:, None]
        ) * TIME_SLOT / MINUTES_PER_HOUR * 1e-3

        return costs

    def _run(self, state: tf.Tensor,
             training: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        buying_price, injection_price = self.grid.take_decision(state)
        p2p_price = (buying_price + injection_price) / 2
        p2p_power = tf.zeros((len(self.agents), len(self.agents)))

        power_exchanges = tf.TensorArray(dtype=tf.float32, size=len(self.agents), dynamic_size=False)

        for r in range(self._rounds + 1):
            p2p_power = p2p_power - tf.linalg.tensor_diag(tf.linalg.tensor_diag_part(p2p_power))

            for i, agent in enumerate(self.agents):
                if training:
                    # Run the model and to get action probabilities and critic value
                    action, _, _ = agent(tf.expand_dims(state, axis=0),
                                         p2p_power[:, i])
                else:
                    action, _, q = agent.take_decision(tf.expand_dims(state, axis=0), p2p_power[:, i])
                power_exchanges = power_exchanges.write(i, -action)

            p2p_power = power_exchanges.stack()

        p_grid, p_p2p = self._assign_powers(p2p_power)

        return -p_grid, -p_p2p, buying_price, injection_price, p2p_price

    def run(self) -> Tuple[tf.Tensor, tf.Tensor]:
        len_env = len(env)
        buying_price = tf.TensorArray(dtype=tf.float32, size=len_env, dynamic_size=False)
        injection_price = tf.TensorArray(dtype=tf.float32, size=len_env, dynamic_size=False)
        p2p_price = tf.TensorArray(dtype=tf.float32, size=len_env, dynamic_size=False)
        grid_power = tf.TensorArray(dtype=tf.float32, size=len_env, dynamic_size=False)
        peer_power = tf.TensorArray(dtype=tf.float32, size=len_env, dynamic_size=False)

        for time, (features, next_features) in enumerate(env.data):

            power_grid, power_p2p, price_buy, price_inj, price_p2p = self._run(features)

            grid_power = grid_power.write(time, power_grid)
            peer_power = peer_power.write(time, power_p2p)
            buying_price = buying_price.write(time, price_buy)
            injection_price = injection_price.write(time, price_inj)
            p2p_price = p2p_price.write(time, price_p2p)

            self._step()

        grid_power = grid_power.stack()
        peer_power = peer_power.stack()

        costs = tf.math.reduce_sum(self._compute_costs(grid_power, peer_power,
                                                       buying_price.concat(),
                                                       injection_price.concat(),
                                                       p2p_price.concat()),
                                   axis=0)

        return grid_power + peer_power, costs

    def init_buffers(self) -> None:
        for _ in range(1):
            for t, (state, next_state) in enumerate(env.data):

                power_grid, power_p2p, p_buy, p_inj, p_p2p = self._run(state, training=True)

                costs = tf.squeeze(self._compute_costs(power_grid, power_p2p, tf.expand_dims(p_buy, axis=0),
                                                       tf.expand_dims(p_inj, axis=0), tf.expand_dims(p_p2p, axis=0)))

                for i, agent in enumerate(self.agents):
                    # Compute reward
                    r = agent.get_reward(costs[i])

                    agent.save_memory(r, tf.expand_dims(next_state, axis=0), tf.zeros(len(self.agents)))

                self._step()

            # Reset iterators
            for agent in self.agents:
                agent.reset()

        for agent in self.agents:
            agent.trainer.initialize_target()

    def train_episode(self, all_rewards: tf.TensorArray, all_losses: tf.TensorArray,
                      _rewards: tf.TensorArray, _losses: tf.TensorArray) -> Tuple[float, float]:
        for t, (state, next_state) in enumerate(env.data):

            power_grid, power_p2p, p_buy, p_inj, p_p2p = self._run(state, training=True)

            costs = tf.squeeze(self._compute_costs(power_grid, power_p2p, tf.expand_dims(p_buy, axis=0),
                                                   tf.expand_dims(p_inj, axis=0), tf.expand_dims(p_p2p, axis=0)))

            for i, agent in enumerate(self.agents):
                # Compute reward
                r = agent.get_reward(costs[i])
                l = agent.train(r, tf.expand_dims(next_state, axis=0), tf.zeros(len(self.agents)))

                # Save statistics
                _rewards = _rewards.write(i, r)
                _losses = _losses.write(i, l)

            all_rewards = all_rewards.write(t, _rewards.concat())
            all_losses = all_losses.write(t, _losses.concat())

            self._step()

        # Reset iterators
        for agent in self.agents:
            agent.reset()

        all_rewards = all_rewards.stack()
        all_losses = all_losses.stack()

        avg_reward = float(tf.math.reduce_sum(tf.math.reduce_mean(all_rewards, axis=-1)))
        avg_loss = float(tf.math.reduce_mean(all_losses))

        return avg_reward, avg_loss

    def _step(self) -> None:
        for agent in self.agents:
            agent.step()

        self.grid.step()

    def reset(self) -> None:
        for agent in self.agents:
            agent.reset()

        self.grid.reset()


class SingleAgentCommunity(CommunityMicrogrid):

    def __init__(self, timeline: pd.DataFrame, agents: List[ActingAgent]) -> None:
        super(SingleAgentCommunity, self).__init__(timeline, agents)

        self.agent = agents[0]

    def _compute_costs_individual(self, grid_power: tf.Tensor, buying_price: tf.Tensor,
                                  injection_price: tf.Tensor,) -> tf.Tensor:
        costs = (
            tf.where(grid_power >= 0.,
                     grid_power * buying_price[:, None],
                     grid_power * injection_price[:, None])
        ) * TIME_SLOT / MINUTES_PER_HOUR * 1e-3

        return costs

    def _run(self, state: tf.Tensor,
             training: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        buying_price, injection_price = self.grid.take_decision(state)

        if training:
            # Run the model and to get action probabilities and critic value
            action, _ = self.agent(tf.expand_dims(state, axis=0))
        else:
            action, _ = self.agent.take_decision(tf.expand_dims(state, axis=0))

        return action, None, buying_price, injection_price, None

    def run(self) -> Tuple[tf.Tensor, tf.Tensor]:
        len_env = len(env)
        buying_price = tf.TensorArray(dtype=tf.float32, size=len_env, dynamic_size=False)
        injection_price = tf.TensorArray(dtype=tf.float32, size=len_env, dynamic_size=False)
        grid_power = tf.TensorArray(dtype=tf.float32, size=len_env, dynamic_size=False)

        for time, (features, next_features) in enumerate(env.data):
            power_grid, _, price_buy, price_inj, _ = self._run(features)

            grid_power = grid_power.write(time, power_grid)
            buying_price = buying_price.write(time, price_buy)
            injection_price = injection_price.write(time, price_inj)

            self._step()

        grid_power = grid_power.stack()

        costs = self._compute_costs_individual(grid_power,
                                               buying_price.concat(),
                                               injection_price.concat())

        return grid_power, costs

    def init_buffers(self) -> None:
        while not self.agent.trainer.buffer.full:
            for t, (state, next_state) in enumerate(env.data):

                power_grid, _, p_buy, p_inj, _ = self._run(state, training=True)

                costs = tf.squeeze(self._compute_costs_individual(power_grid,
                                                                  tf.expand_dims(p_buy, axis=0),
                                                                  tf.expand_dims(p_inj, axis=0)))

                # Compute reward
                r = self.agent.get_reward(costs)
                self.agent.save_memory(r, tf.expand_dims(next_state, axis=0))

                self._step()

            # Reset iterators
            self.agent.reset()

        self.agent.trainer.initialize_target()

    def train_episode(self, all_rewards: tf.TensorArray, all_losses: tf.TensorArray,
                      *args, **kwargs) -> Tuple[float, float]:
        for t, (state, next_state) in enumerate(env.data):

            power_grid, _, p_buy, p_inj, _ = self._run(state, training=True)

            costs = tf.squeeze(self._compute_costs_individual(power_grid, tf.expand_dims(p_buy, axis=0),
                                                              tf.expand_dims(p_inj, axis=0)))

            # Compute reward
            r = self.agent.get_reward(costs)
            l = self.agent.train(r, tf.expand_dims(next_state, axis=0))

            # Save statistics
            all_rewards = all_rewards.write(t, r)
            all_losses = all_losses.write(t, l)

            self._step()

        self.agent.reset()

        all_rewards = all_rewards.stack()
        all_losses = all_losses.stack()

        total_reward = float(tf.math.reduce_sum(all_rewards))
        avg_loss = float(tf.math.reduce_mean(all_losses))

        return total_reward, avg_loss


def main(con: sqlite3.Connection, load_agents: bool = False, analyse: bool = False) -> None:

    print("Creating community...")
    community = get_rl_based_community(nr_agents)
    if load_agents:
        community.agent.load_from_file(setting, implementation)

    env_len = len(env)
    rewards = tf.TensorArray(dtype=tf.float32, size=env_len, dynamic_size=False)
    losses = tf.TensorArray(dtype=tf.float32, size=env_len, dynamic_size=False)

    if implementation == 'dqn':
        print("Initializing buffers...")
        community.init_buffers()

    print("Training...")
    time_start_training = time.time()
    with trange(starting_episode, max_episodes) as episodes:
        for episode in episodes:
            reward, error = community.train_episode(rewards, losses, None, None)

            episodes_reward.append(reward)
            episodes_error.append(error)

            if episode % min_episodes_criterion == 0:
                _reward = statistics.mean(episodes_reward)
                _error = statistics.mean(episodes_error)
                print(f'Average reward: {_reward:.3f}. '
                      f'Average error: {_error:.3f}')

                db.log_training_progress(con, setting, implementation, episode, _reward, _error)
                community.agent.actor.decay_exploration()

            if (episode + 1) % save_episodes == 0:
                community.agent.save_to_file(setting, implementation)

        _reward = statistics.mean(episodes_reward)
        _error = statistics.mean(episodes_error)
        db.log_training_progress(con, setting, implementation, episode, _reward, _error)
        community.agent.save_to_file(setting, implementation)

    time_end_training = time.time()

    if analyse:
        print("Running...")
        env_df, agent_df = ds.get_validation_data()
        env.setup(ds.dataframe_to_dataset(env_df))
        for agent in community.agents:
            agent_load = ds.dataframe_to_dataset(agent_df['l0'] * 0.7 * 1e3)
            agent_pv = ds.dataframe_to_dataset(agent_df['pv'] * 4 * 1e3)
            agent.set_profiles(agent_load, agent_pv)

        time_start_run = time.time()
        power, cost = community.run()
        time_end_run = time.time()
        cost = tf.math.reduce_sum(cost, axis=0)

        print("Analysing...")
        save_times(train_time=time_end_training - time_start_training, run_time=time_end_run - time_start_run)
        analyse_community_output(community.agents, community.timeline.tolist(), power.numpy(), cost.numpy())


def save_times(train_time: Optional[float] = None, run_time: Optional[float] = None) -> None:
    data: dict
    with open('../data/timing_data.json', 'r') as data_file:
        data = json.load(data_file)

    if setting in data:
        if implementation not in data[setting]:
            data[setting][implementation] = {}

        if train_time:
            data[setting][implementation]['train'] = train_time
        if run_time:
            data[setting][implementation]['run'] = run_time
    else:
        data[setting] = {}
        data[setting][implementation] = {'train': train_time, 'run': run_time}

    with open('../data/timing_data.json', 'w') as data_file:
        json.dump(data, data_file)


def save_community_results(con: sqlite3.Connection, is_testing: bool,
                           setting: str, day: int,
                           community: SingleAgentCommunity, cost: np.ndarray) -> None:
    time = [float(state[0]) for state, _ in env.data]
    loads = list(map(lambda l: float(l[0]), community.agent._load))
    pvs = community.agent.pv.get_history()
    temperatures = community.agent.heating.get_history()
    heatpump = community.agent.heating._power_history
    costs = cost[:, 0].tolist()
    days = [day] * len(time)

    if is_testing:
        db.log_test_results(con, setting, 0, days, time, loads, pvs, temperatures, heatpump, costs, implementation)
    else:
        db.log_validation_results(con, setting, 0, days, time, loads, pvs, temperatures, heatpump,
                                  costs, implementation)


def load_and_run(con: Optional[sqlite3.Connection] = None, is_testing: bool = False,
                 analyse: bool = True) -> None:

    print("Creating community...")
    if implementation == 'rule-based':
        community = get_rule_based_community(nr_agents)
    elif implementation == 'semi-intelligent':
        community = get_baseline_community()
    elif implementation == 'tabular':
        community = get_rl_based_community(nr_agents)
        community.agent.load_from_file(setting, implementation)
    else:
        raise RuntimeError(f"Unknown impplementation: {implementation}")

    env_df, agent_df = ds.get_test_data() if is_testing else ds.get_validation_data()
    days = np.unique(env_df['day'])
    day_indices = {day: env_df['day'] == day for day in days}
    env_df.drop(axis=1, labels='day', inplace=True)

    # Run from fresh start for each day (don't propagate bad decisions)
    for day in days:
        print(f'Running day {day}')
        env.setup(ds.dataframe_to_dataset(env_df[day_indices[day]]))
        community.reset()

        for agent in community.agents:
            agent_load = ds.dataframe_to_dataset(agent_df.loc[day_indices[day], 'l0'] * 0.7 * 1e3)
            agent_pv = ds.dataframe_to_dataset(agent_df.loc[day_indices[day], 'pv'] * 4 * 1e3)
            agent.set_profiles(agent_load, agent_pv)

        print("Running...")
        power, cost = community.run()

        if con:
            print("Saving...")
            save_community_results(con, is_testing, setting, day, community, cost.numpy())

    if analyse:
        cost = tf.math.reduce_sum(cost, axis=0)

        print("Analysing...")
        analyse_community_output(community.agents, community.timeline.tolist(), power.numpy(), cost.numpy())


starting_episode = 0
max_episodes = 1000
min_episodes_criterion = 50
save_episodes = 50
nr_agents = 1
setting = 'single-agent'
implementation = 'tabular'

episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
episodes_error: collections.deque = collections.deque(maxlen=min_episodes_criterion)


if __name__ == '__main__':

    db_connection = db.get_connection()

    try:
        # main(db_connection, analyse=True)
        load_and_run(db_connection, is_testing=True, analyse=False)
    finally:
        if db_connection:
            db_connection.close()

