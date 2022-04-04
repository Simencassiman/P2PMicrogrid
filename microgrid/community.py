# Python Libraries
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from datetime import datetime

import pandas as pd
import tensorflow as tf

# Local Modules
from microgrid.environment import env
from config import TIME_SLOT, MINUTES_PER_HOUR, HOURS_PER_DAY
from agent import Agent, ActingAgent, GridAgent, RuleAgent
from production import Prosumer, Consumer, PV
from storage import BatteryStorage, NoStorage, Battery
from heating import HPHeating, HeatPump
from data_analysis import analyse_community_output
import database as db
import dataset as ds


def get_rule_based_community(n_agents: int = 5) -> CommunityMicrogrid:

    # conn = db.get_connection(DB_PATH)

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
    # load = np.array(data['l0']) / data['l0'].max()
    # production = np.array(data['pv']) / data['pv'].max()
    # temperature = np.array(data['temperature'])
    # cloud_cover = np.array(data['cloud_cover'])
    # humidity = np.array(data['humidity'])
    # irradiance = 1.7 * np.ones(temperature.shape)

    agents: List[ActingAgent] = []

    load_ratings = np.random.normal(0.7, 0.2, n_agents)
    pv_ratings = np.random.normal(4, 0.2, n_agents)

    # Create agents
    Agent.reset_ids()
    for i in range(n_agents):

        max_power = 1e3
        safety = 1e3

        agent_load = ds.dataframe_to_dataset(agent_df['l0'] * (load_ratings[i] if i < 3 else load_ratings[0]) * 1e3)
        agent_pv = ds.dataframe_to_dataset(agent_df['pv'] * pv_ratings[i] * 1e3)

        agents.append(RuleAgent(agent_load,
                                Prosumer(PV(peak_power=pv_ratings[i] * 1e3,
                                            production=agent_pv)) if i < 4 else Consumer(),
                                NoStorage(),
                                HPHeating(HeatPump(cop=3.0, max_power=3 * 1e3, power=0.0), 21.0),
                                max_in=max_power + safety,
                                max_out=-(max_power + safety)
                                )
                      )

    # Prepare environment
    env.setup(ds.dataframe_to_dataset(env_df))

    return CommunityMicrogrid(timeline, agents)

class CommunityMicrogrid:

    def __init__(self, timeline: pd.DataFrame, agents: List[ActingAgent]) -> None:
        self.timeline = timeline
        self.time_length = len(timeline)
        self.agents = agents
        self.grid = GridAgent()

    def run(self) -> Tuple[tf.Tensor, tf.Tensor]:

        buying_price = np.zeros(len(env))
        injection_price = np.zeros(len(env))
        power = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for time, (features, next_features) in enumerate(env.data):
            for i, agent in enumerate(self.agents):
                p, _ = agent.take_decision(features)
                power = power.write(time, p)

            buying_price[time], injection_price[time] = self.grid.take_decision(features)

            self._step()

        power = power.stack()

        costs = tf.math.reduce_sum(tf.where(power >= 0,
                                            power * buying_price[:, None],
                                            power * injection_price[:, None]), axis=0) \
                * TIME_SLOT / MINUTES_PER_HOUR * 1e-3
        # costs = (power * TIME_SLOT / 60 * 1e-3 *
        #          ((power >= 0) * buying_price[:, None] - (power < 0) * injection_price[:, None]))

        return power, costs

    def train_episode(self) -> None:
        rewards = []
        # for episode in range(EPISODES):
        #
        #     state = environment.reset()
        #     for _ in range(MAX_STEPS):
        #
        #         if np.random.uniform(0, 1) < epsilon:
        #             action = environment.action_space.sample()
        #         else:
        #             action = np.argmax(Q[state, :])
        #
        #         next_state, reward, done, _ = environment.step(action)
        #
        #         Q[state, action] = Q[state, action] + LEARNING_RATE * (
        #                     reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])
        #
        #         state = next_state
        #
        #         if done:
        #             rewards.append(reward)
        #             epsilon -= 0.001
        #             break  # reached goal
        #
        # print(Q)
        print(f"Average reward: {sum(rewards) / len(rewards)}:")

    def _step(self) -> None:
        pass

    def _iterate(self) -> None:
        pass

    def reset(self) -> None:
        for agent in self.agents:
            agent.reset()

        self.grid.reset()


if __name__ == '__main__':
    nr_agents = 5
    start = datetime(2021, 11, 1)
    end = datetime(2021, 12, 1)

    community = get_rule_based_community(nr_agents)

    power, cost = community.run()

    analyse_community_output(community.agents, community.timeline.tolist(), power.numpy(), cost.numpy())

