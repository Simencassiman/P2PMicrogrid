# Python Libraries
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Local Modules
from microgrid import environment, SECONDS_PER_HOUR
from config import TIME_SLOT, DB_PATH
from agent import Agent, ActingAgent, GridAgent, RuleAgent
from production import Prosumer, Consumer, PV
from storage import BatteryStorage, NoStorage, Battery
from heating import HPHeating, HeatPump
from data_analysis import analyse_community_output
import database as db

def get_rule_based_community(start: datetime, end: datetime, n_agents: int = 5) -> CommunityMicrogrid:

    conn = db.get_connection(DB_PATH)

    # Load time series
    data = db.get_data(conn, start, end)
    data['isodate'] = data['date'] + ' ' + data['time'] + data['utc']
    max_load = 3 * data['l0'].median()
    data.loc[data['l0'].abs() >= max_load, 'l0'] = max_load

    timeline = np.array([datetime.fromisoformat(date) for date in data['isodate']])
    load = np.array(data['l0']) / data['l0'].max()
    production = np.array(data['pv']) / data['pv'].max()
    temperature = np.array(data['temperature'])
    cloud_cover = np.array(data['cloud_cover'])
    humidity = np.array(data['humidity'])
    irradiance = 1.7 * np.ones(temperature.shape)
    agents: List[ActingAgent] = []

    load_ratings = np.random.normal(0.7, 0.2, n_agents)
    pv_ratings = np.random.normal(4, 0.2, n_agents)

    # Create agents
    Agent.reset_ids()
    for i in range(n_agents):

        max_power = 1e3
        safety = 1e3

        agents.append(RuleAgent(load * (load_ratings[i] if i < 3 else load_ratings[0]) * 1e3,
                                Prosumer(PV(peak_power=pv_ratings[i] * 1e3,
                                            production=production * pv_ratings[i] * 1e3)) if i < 4 else Consumer(),
                                BatteryStorage(Battery(capacity=7800 * SECONDS_PER_HOUR, peak_power=5000, min_soc=0.2,
                                                       max_soc=0.8, efficiency=0.9, soc=0.5)) if i < 3 else NoStorage(),
                                HPHeating(HeatPump(cop=3.0, max_power=3000, power=0.0), 21.0),
                                max_in=max_power + safety,
                                max_out=-(max_power + safety)
                                )
                      )

    # Prepare environment
    environment.setup(temperature, cloud_cover, humidity, irradiance)

    return CommunityMicrogrid(timeline, agents)

class CommunityMicrogrid:

    def __init__(self, timeline: np.ndarray, agents: List[ActingAgent]):
        self.timeline = timeline
        self.time_length = len(timeline)
        self.agents = agents
        self.grid = GridAgent()

    def run(self) -> Tuple[np.ndarray, np.ndarray]:

        buying_price = np.zeros(self.time_length)
        injection_price = np.zeros(self.time_length)
        power = np.zeros((self.time_length, len(self.agents)))

        for time in range(self.time_length):
            for i, agent in enumerate(self.agents):
                power[time, i], _ = agent.take_decision()

            buying_price[time], injection_price[time] = self.grid.take_decision()

            self._step()

        costs = (power * TIME_SLOT / 60 * 1e-3 *
                 ((power >= 0) * buying_price[:, None] - (power < 0) * injection_price[:, None]))

        return power, costs

    def train(self) -> None:
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

    community = get_rule_based_community(start, end, nr_agents)

    power, cost = community.run()

    analyse_community_output(community.agents, community.timeline.tolist(), power, cost)

