# Python Libraries
from __future__ import annotations

import collections
import gc
import statistics
from typing import List, Tuple, Callable, Any
import numpy as np
from datetime import datetime

import pandas as pd
import tensorflow as tf
from tqdm import trange
import matplotlib.pyplot as plt

# Local Modules
from environment import env
from config import TIME_SLOT, MINUTES_PER_HOUR, HOURS_PER_DAY
from agent import Agent, ActingAgent, GridAgent, RuleAgent, RLAgent
from production import Prosumer, Consumer, PV
from storage import NoStorage
from heating import HPHeating, HeatPump
from data_analysis import analyse_community_output
import dataset as ds


def get_community(agent_constructor: Callable[[Any], ActingAgent], n_agents: int) -> CommunityMicrogrid:
    # Load time series
    env_df, agent_df = ds.get_train_data()

    # Get dates for plotting
    # ds.get_train_isodate()
    # data = db.get_data(conn, start, end)
    # data['isodate'] = data['date'] + ' ' + data['time'] + data['utc']

    # Take care of outliers, should do this on actual data
    # max_load = 3 * data['l0'].median()
    # data.loc[data['l0'].abs() >= max_load, 'l0'] = max_load

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

        agent_load = ds.dataframe_to_dataset(agent_df['l0'] * (load_ratings[i] if i < 3 else load_ratings[0]) * 1e3)
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

    return CommunityMicrogrid(timeline, agents)


def get_rule_based_community(n_agents: int = 5) -> CommunityMicrogrid:
    return get_community(RuleAgent, n_agents)


def get_rl_based_community(n_agents: int = 5) -> CommunityMicrogrid:
    return get_community(RLAgent, n_agents)


class CommunityMicrogrid:

    def __init__(self, timeline: pd.DataFrame, agents: List[ActingAgent]) -> None:
        self.timeline = timeline
        self.time_length = len(timeline)
        self.agents = agents
        self.grid = GridAgent()
        self.q = np.zeros((len(env), len(agents), 3))

    def run(self) -> Tuple[tf.Tensor, tf.Tensor]:

        buying_price = np.zeros(len(env))
        injection_price = np.zeros(len(env))
        power = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for time, (features, next_features) in enumerate(env.data):
            for i, agent in enumerate(self.agents):
                p, _, self.q[time, i, :] = agent.take_decision(tf.expand_dims(features, axis=0))
                power = power.write(time, p)

            buying_price[time], injection_price[time] = self.grid.take_decision(features)

            self._step()

        power = power.stack()

        costs = tf.math.reduce_sum(tf.where(power >= 0,
                                            power * buying_price[:, None],
                                            power * injection_price[:, None]), axis=0) \
                * TIME_SLOT / MINUTES_PER_HOUR * 1e-3

        return power, costs

    def init_buffers(self) -> None:
        for _ in range(10):
            for t, (state, next_state) in enumerate(env.data):
                p_buy, p_inj = self.grid.take_decision(state)

                for agent in self.agents:
                    # Run the model and to get action probabilities and critic value
                    action, _, _ = agent.explore(tf.expand_dims(state[:1], axis=0))

                    # Compute reward
                    r = agent.get_reward(action[0] / 1e3, p_buy, p_inj)

                    agent.save_memory(r, tf.expand_dims(next_state, axis=0))

                self._step()

            # Reset iterators
            for agent in self.agents:
                agent.reset()

        for agent in self.agents:
            agent.trainer.initialize_target()

    def train_episode(self) -> Tuple[float, float]:
        rewards = np.zeros((len(env), len(self.agents)))
        losses = np.zeros(rewards.shape)

        for t, (state, next_state) in enumerate(env.data):
            p_buy, p_inj = self.grid.take_decision(state)

            for i, agent in enumerate(self.agents):
                # Run the model and to get action probabilities and critic value
                action, _, _ = agent(tf.expand_dims(state, axis=0))

                # Compute reward
                r = agent.get_reward(action[0] / 1e3, p_buy, p_inj)

                # Save statistics
                rewards[t, i] = r.numpy()
                losses[t, i] = agent.train(r, tf.expand_dims(next_state, axis=0))

            self._step()

        # Reset iterators
        for agent in self.agents:
            agent.reset()

        # print(f"Average reward: {sum(rewards) / len(rewards)}:")
        return rewards.mean(axis=-1).sum(), losses.mean()

    def _step(self) -> None:
        for agent in self.agents:
            agent.step()
        self.grid.step()

    def _iterate(self) -> None:
        pass

    def reset(self) -> None:
        for agent in self.agents:
            agent.reset()

        self.grid.reset()


max_episodes = 5 * 1000
min_episodes_criterion = 100

episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
episodes_error: collections.deque = collections.deque(maxlen=min_episodes_criterion)


if __name__ == '__main__':
    nr_agents = 1
    start = datetime(2021, 11, 1)
    end = datetime(2021, 12, 1)

    community = get_rl_based_community(nr_agents)

    community.init_buffers()

    with trange(max_episodes) as episodes:
        for episode in episodes:
            reward, error = community.train_episode()
            episodes_reward.append(reward)
            episodes_error.append(error)

            if episode % min_episodes_criterion == 0:
                print(f'Average reward: {statistics.mean(episodes_reward):.3f}. '
                      f'Average error: {statistics.mean(episodes_error):.3f}')

                for agent in community.agents:
                    agent.actor.decay_exploration()

    power, cost = community.run()

    plt.figure(0)
    plt.plot(np.arange(community.q.shape[0]), community.q[:, 0, :])
    plt.legend(['0', '1', '2'])
    analyse_community_output(community.agents, community.timeline.tolist(), power.numpy(), cost.numpy())

