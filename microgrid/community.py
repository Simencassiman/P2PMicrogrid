# Python Libraries
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Local Modules
from microgrid import environment
from config import TIME_SLOT
from agent import Agent, ActingAgent, GridAgent, RuleAgent
from production import Prosumer, Consumer, PV
from storage import BatteryStorage, NoStorage, Battery
from heating import HPHeating, HeatPump
from data_analysis import analyse_community_output

def get_rule_based_community(n_agents: int = 5) -> CommunityMicrogrid:

    timeline: np.ndarray
    load: npt.ArrayLike[float]
    temperature: npt.ArrayLike[float]
    cloud_cover: npt.ArrayLike[float]
    humidity: npt.ArrayLike[float]
    # irradiance: npt.ArrayLike[float]
    production: npt.ArrayLike[float]
    agents: List[ActingAgent] = []

    with open(f'../data/profiles.json') as file:
        data = json.load(file)

        timeline = np.array([datetime.fromisoformat(date) for date in data['time']])
        load = np.array(data['loads'])
        production = np.array(data['pv'])
        temperature = np.array(data['temperature'])
        cloud_cover = np.array(data['cloud_cover'])
        humidity = np.array(data['humidity'])
        # irradiance = 1.7 * np.ones(temperature.shape)

    load_ratings = np.random.normal(0.7, 0.2, n_agents)
    pv_ratings = np.random.normal(3, 0.2, n_agents)

    Agent.reset_ids()
    for i in range(n_agents):

        max_power = 1e3
        safety = 1e3

        agents.append(RuleAgent(load * (load_ratings[i] if i < 3 else load_ratings[0]) * 1e3,
                                Prosumer(PV(peak_power=pv_ratings[i] * 1e3,
                                            production=production * pv_ratings[i] * 1e3)) if i < 4 else Consumer(),
                                BatteryStorage(Battery(capacity=7800 * 3600, peak_power=5000, min_soc=0.2, max_soc=0.8,
                                                       efficiency=0.9, soc=0.5)) if i < 3 else NoStorage(),
                                HPHeating(HeatPump(cop=3.0, max_power=3000, power=0.0), 21.0),
                                max_in=max_power + safety,
                                max_out=-(max_power + safety)
                                )
                      )

    environment.setup(temperature, cloud_cover, humidity)

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
        pass

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

    with open('../data/profiles.json') as file:
        time = json.load(file)['time']

    community = get_rule_based_community(nr_agents)

    power, cost = community.run()

    analyse_community_output(community.agents, time, power, cost)

