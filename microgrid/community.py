# Python Libraries
from __future__ import annotations
import traceback
import collections
import sqlite3
import statistics
from typing import List, Tuple, Callable, Any, Optional
import numpy as np
import re

import pandas as pd
import tensorflow as tf
from tqdm import trange
import matplotlib.pyplot as plt

# Local Modules
from environment import env
from config import TIME_SLOT, MINUTES_PER_HOUR, HOURS_PER_DAY
from agent import Agent, ActingAgent, GridAgent, RuleAgent, QAgent
from production import Prosumer, PV
from storage import NoStorage
from heating import HPHeating, HeatPump
from data_analysis import analyse_community_output
import dataset as ds
import database as db


def get_community(agent_constructor: Callable[[Any], ActingAgent], n_agents: int,
                  homogeneous: bool = False) -> CommunityMicrogrid:
    # Load time series
    env_df, agent_dfs = ds.get_train_data()

    if homogeneous:
        agent_dfs = [agent_dfs[0]] * n_agents

    timeline = env_df['time'].map(lambda t: int(t * MINUTES_PER_HOUR / TIME_SLOT * HOURS_PER_DAY))
    # timeline = np.array([datetime.fromisoformat(date) for date in data['isodate']])

    agents: List[ActingAgent] = []

    load_ratings = np.random.normal(0.7, 0.2, n_agents) if not homogeneous else np.array([0.7] * n_agents)
    pv_ratings = np.random.normal(4, 0.2, n_agents) if not homogeneous else np.array([4] * n_agents)

    # Create agents
    Agent.reset_ids()
    for i in range(n_agents):
        max_power = max(load_ratings[i], pv_ratings[i])
        safety = 1.1

        agent_load = ds.dataframe_to_dataset(agent_dfs[i]['load'] * load_ratings[i] * 1e3)
        agent_pv = ds.dataframe_to_dataset(agent_dfs[i]['pv'] * pv_ratings[i] * 1e3)

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

    return CommunityMicrogrid(timeline, agents)


def get_rule_based_community(n_agents: int = 5) -> CommunityMicrogrid:
    return get_community(RuleAgent, n_agents)


def get_rl_based_community(n_agents: int, homogeneous: bool) -> CommunityMicrogrid:
    return get_community(QAgent, n_agents, homogeneous)


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
                           p2p_power,
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
                    action, _ = agent(tf.expand_dims(state, axis=0), -p2p_power[:, i])
                else:
                    action, q = agent.take_decision(tf.expand_dims(state, axis=0), -p2p_power[:, i])
                power_exchanges = power_exchanges.write(i, action)

            p2p_power = power_exchanges.stack()

        p_grid, p_p2p = self._assign_powers(p2p_power)

        return p_grid, p_p2p, buying_price, injection_price, p2p_price

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

        costs = self._compute_costs(grid_power, peer_power,
                                    buying_price.concat(),
                                    injection_price.concat(),
                                    p2p_price.concat())

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

                    agent.save_memory(r, tf.expand_dims(next_state, axis=0))

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
                l = agent.train(r, tf.expand_dims(next_state, axis=0))

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


def main(con: sqlite3.Connection, load_agent: bool = False, analyse: bool = True) -> None:

    print("Creating community...")
    community = get_rl_based_community(nr_agents, homogeneous=homogeneous)

    if load_agent:
        for i, agent in enumerate(community.agents):
            agent.actor.set_qtable(np.load(f'../models_{implementation}/{re.sub("-", "_", setting)}_{i}.npy'))

    env_len = len(env)
    agents_len = len(community.agents)
    rewards = tf.TensorArray(dtype=tf.float32, size=env_len, dynamic_size=False)
    losses = tf.TensorArray(dtype=tf.float32, size=env_len, dynamic_size=False)
    _rewards = tf.TensorArray(dtype=tf.float32, size=agents_len, dynamic_size=False)
    _losses = tf.TensorArray(dtype=tf.float32, size=agents_len, dynamic_size=False)

    # print("Initializing buffers...")
    # community.init_buffers()

    print("Training...")
    with trange(starting_episodes, max_episodes) as episodes:
        for episode in episodes:
            reward, error = community.train_episode(rewards, losses, _rewards, _losses)

            episodes_reward.append(reward)
            episodes_error.append(error)

            if episode % min_episodes_criterion == 0:
                _reward = statistics.mean(episodes_reward)
                _error = statistics.mean(episodes_error)
                print(f'Average reward: {_reward:.3f}. '
                      f'Average error: {_error:.3f}')

                for agent in community.agents:
                    agent.actor.decay_exploration()

                db.log_training_progress(con, setting, implementation, episode, _reward, _error)

            if (episode + 1) % save_episodes == 0:
                for i, agent in enumerate(community.agents):
                    np.save(f'../models_{implementation}/{re.sub("-", "_", setting)}_{i}.npy', agent.actor.q_table)

        _reward = statistics.mean(episodes_reward)
        _error = statistics.mean(episodes_error)
        db.log_training_progress(con, setting, implementation, episode, _reward, _error)
        for i, agent in enumerate(community.agents):
            np.save(f'../models_{implementation}/{re.sub("-", "_", setting)}_{i}.npy', agent.actor.q_table)

    if analyse:
        print("Running...")
        env_df, agent_dfs = ds.get_validation_data()
        env.setup(ds.dataframe_to_dataset(env_df))
        for i, agent in enumerate(community.agents):
            agent_load = ds.dataframe_to_dataset(agent_dfs[i]['load'] *
                                                 (0.7e3 if homogeneous else np.random.normal(0.7, 0.2, 1) * 1e3))
            agent_pv = ds.dataframe_to_dataset(agent_dfs[i]['pv'] *
                                               (4e3 if homogeneous else np.random.normal(4, 0.2, 1) * 1e3))
            agent.set_profiles(agent_load, agent_pv)

        power, cost = community.run()

        print("Analysing...")
        analyse_community_output(community.agents, community.timeline.tolist(), power.numpy(), cost.numpy())


def save_community_results(con: sqlite3.Connection, is_testing: bool,
                           setting: str, days: List[int],
                           community: CommunityMicrogrid, cost: np.ndarray) -> None:
    time = [float(state[0]) for state, _ in env.data]
    loads = [list(map(lambda l: float(l[0]), agent._load)) for agent in community.agents]
    pvs = [agent.pv.get_history() for agent in community.agents]
    temperatures = [agent.heating.get_history() for agent in community.agents]
    heatpump = [agent.heating._power_history for agent in community.agents]
    costs = [cost[:, i].tolist() for i in range(cost.shape[-1])]

    for i, data in enumerate(zip(loads, pvs, temperatures, heatpump, costs)):
        if is_testing:
            db.log_test_results(con, setting, i, days, time, *data, implementation)
        else:
            db.log_validation_results(con, setting, i, days, time, *data, implementation)


def load_and_run(con: Optional[sqlite3.Connection] = None, is_testing: bool = False) -> None:

    print("Creating community...")
    community = get_rl_based_community(nr_agents, homogeneous=homogeneous)

    env_df, agent_dfs = ds.get_test_data() if is_testing else ds.get_validation_data()
    days = env_df['day'].tolist()
    env_df.drop(axis=1, labels='day', inplace=True)
    if homogeneous:
        agent_dfs = [agent_dfs[0]] * nr_agents

    env.setup(ds.dataframe_to_dataset(env_df))
    for i, agent in enumerate(community.agents):
        agent_load = ds.dataframe_to_dataset(agent_dfs[i]['load'] *
                                             (0.7e3 if homogeneous else np.random.normal(0.7, 0.2, 1) * 1e3))
        agent_pv = ds.dataframe_to_dataset(agent_dfs[i]['pv'] *
                                           (4e3 if homogeneous else np.random.normal(4, 0.2, 1) * 1e3))
        agent.set_profiles(agent_load, agent_pv)
        agent.actor.set_qtable(np.load(f'../models_{implementation}/{re.sub("-", "_", setting)}_{i}.npy'))

    print("Running...")
    power, cost = community.run()

    if con:
        print("Saving...")
        save_community_results(con, is_testing, setting, days, community, cost.numpy())

    cost = tf.reduce_sum(cost, axis=0)

    print("Analysing...")
    analyse_community_output(community.agents, community.timeline.tolist(), power.numpy(), cost.numpy())


starting_episodes = 151
max_episodes = 1000
min_episodes_criterion = 50
save_episodes = 50
nr_agents = 2
rounds = 1
homogeneous = False
setting = f'{nr_agents}-multi-agent-rounds-{rounds}-no-com-{"homo" if homogeneous else "hetero"}'
implementation = 'tabular'


episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
episodes_error: collections.deque = collections.deque(maxlen=min_episodes_criterion)


if __name__ == '__main__':

    db_connection = db.get_connection()

    try:
        # main(db_connection, load_agent=True, analyse=True)
        load_and_run(db_connection, is_testing=True)
    finally:
        if db_connection:
            db_connection.close()



