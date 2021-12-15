# Python Libraries
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Local Modules
from __init__ import environment
from config import TIME_SLOT
from agent import ActingAgent, GridAgent, RuleAgent
from production import Prosumer, PV
from storage import BatteryStorage, Battery
from heating import HPHeating, HeatPump

def get_rule_based_community(n_agents: int = 5) -> CommunityMicrogrid:

    timeline: np.ndarray
    load: npt.ArrayLike[float]
    temperature: npt.ArrayLike[float]
    cloud_cover: npt.ArrayLike[float]
    humidity: npt.ArrayLike[float]
    irradiance: npt.ArrayLike[float]
    production: npt.ArrayLike[float]
    agents: List[ActingAgent] = []

    with open(f'data/profiles.json') as file:
        data = json.load(file)

        timeline = np.array([datetime.fromisoformat(date) for date in data['time']])
        load = np.array(data['loads'])
        print(load[0])
        temperature = np.array(data['temperature'])
        cloud_cover = np.array(data['cloud_cover'])
        humidity = np.array(data['humidity'])
        irradiance = 1.7 * np.ones(temperature.shape)

    ratings = np.random.normal(0.7, 0.1, n_agents)

    for i in range(n_agents):

        pv = np.zeros(temperature.shape)
        max_pv = 0
        max_power = 1
        safety = 1e3

        agents.append(RuleAgent(load * ratings[i] * 1e3,
                                Prosumer(PV(peak_power=max_pv, production=pv)),
                                BatteryStorage(Battery(capacity=7800, peak_power=5000, min_soc=0.2, max_soc=0.8,
                                                       efficiency=0.9, soc=0.5)),
                                HPHeating(HeatPump(cop=3.0, max_power=3000, power=0.0), 21.0),
                                max_in=max_power + safety,
                                max_out=-(max_power + safety)
                                )
                      )

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
    community = get_rule_based_community(2)

    power, cost = community.run()
    print(f'Energy consumed: {power.sum(axis=0) * TIME_SLOT / 60 * 1e-3} kWh')
    print(f'Cost a total of: {cost.sum(axis=0)} €')
    print('Grid load')
    print(power)

    temp = community.agents[0].heating._history
    t = [i for i in range(len(temp))]

    plt.plot(t, temp)
    plt.show()
