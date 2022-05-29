# Python Libraries
from __future__ import annotations
from typing import List, Tuple, TypeVar
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Local modules
import config as cf
from config import TIME_SLOT, SECONDS_PER_MINUTE
from environment import env
from electrical_asset import ElectricalAsset
import dataset as ds


np.random.seed(42)

"""
    Simulator parameters
"""

Ci = 2.44e6 * 2
Cm = 9.4e7
Ri = 8.64e-4
Re = 1.05e-2
Rvent = 7.98e-3
gA = 11.468
f_rad = 0.3

"""
    End parameters
"""


T = TypeVar('T')
def temperature_simulation(t_out: float, t_in: float, t_bm: float,
                           hp_power: float, hp_cop: float,
                           solar_rad: float = 0) -> Tuple[float, float]:
    dT_in = 1 / Ci * (
            1 / Ri * (t_bm - t_in)
            + 1 / Rvent * (t_out - t_in)
            + (1 - f_rad) * hp_power * hp_cop
    )

    dT_m = 1 / Cm * (
            1 / Ri * (t_in - t_bm)
            + 1 / Re * (t_out - t_bm)
            + gA * solar_rad
            + f_rad * hp_power * hp_cop
    )

    t_in_new = t_in + dT_in * SECONDS_PER_MINUTE * TIME_SLOT
    t_m_new = t_bm + dT_m * SECONDS_PER_MINUTE * TIME_SLOT

    return t_in_new, t_m_new


class Heating(ElectricalAsset):

    @property
    @abstractmethod
    def lower_bound(self) -> float:
        ...

    @property
    @abstractmethod
    def upper_bound(self) -> float:
        ...

    @property
    @abstractmethod
    def temperature(self) -> tf.Tensor:
        ...

    @property
    @abstractmethod
    def normalized_temperature(self) -> tf.Tensor: ...

    @property
    @abstractmethod
    def power(self) -> tf.Tensor:
        ...

    @abstractmethod
    def has_heater(self) -> bool:
        ...

    @abstractmethod
    def set_power(self, power: float) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...


class HPHeating(Heating):

    TEMPERATURE_MARGIN = 1.

    def __init__(self, hp: HeatPump, temperature_setpoint: float):
        self.temperature_choice = (temperature_setpoint - self.TEMPERATURE_MARGIN,
                                   temperature_setpoint + self.TEMPERATURE_MARGIN)
        self.hp = hp
        self._time = 0
        self._history: List[float] = []
        self._power_history = []

        self._temperature_setpoint = temperature_setpoint
        self._t_building_mass = tf.convert_to_tensor([temperature_setpoint]) if cf.homogeneous \
            else tf.constant(np.random.normal(temperature_setpoint, 0.3, 1), dtype=tf.float32)
        self._t_indoor = tf.convert_to_tensor([temperature_setpoint]) if cf.homogeneous \
            else tf.constant(np.random.normal(temperature_setpoint, 0.3, 1), dtype=tf.float32)

    @property
    def lower_bound(self) -> float:
        return self.temperature_choice[0]

    @property
    def upper_bound(self) -> float:
        return self.temperature_choice[1]

    @property
    def temperature(self) -> tf.Tensor:
        return self._t_indoor

    @property
    def normalized_temperature(self) -> tf.Tensor:
        return (self._t_indoor - self._temperature_setpoint) / self.TEMPERATURE_MARGIN

    @property
    def power(self) -> tf.Tensor:
        return tf.convert_to_tensor([self.hp.power * self.hp.max_power], dtype=tf.float32)

    def get_temperature(self) -> Tuple[tf.Tensor, tf.Tensor]:
        t_out = env.temperature

        return temperature_simulation(t_out, self._t_indoor, self._t_building_mass,
                                      self.power, self.hp.cop)

    def has_heater(self) -> bool:
        return True

    def set_power(self, power: float) -> None:
        self.hp.power = power

    def step(self) -> None:
        self._history.append(float(self._t_indoor[0]))
        self._power_history.append(float(self.power[0]))
        self._t_indoor, self._t_building_mass = self.get_temperature()
        self._t_indoor = self._t_indoor
        self._time += 1

    def reset(self) -> None:
        self._time = 0
        self._history = []
        self._power_history = []
        self._t_indoor = tf.convert_to_tensor([self._temperature_setpoint]) if cf.homogeneous \
            else tf.constant(np.random.normal(self._temperature_setpoint, 0.3, 1), dtype=tf.float32)
        self._t_building_mass = tf.convert_to_tensor([self._temperature_setpoint]) if cf.homogeneous \
            else tf.constant(np.random.normal(self._temperature_setpoint, 0.3, 1), dtype=tf.float32)

    def get_history(self) -> List[float]:
        return self._history


@dataclass
class HeatPump:

    cop: float
    max_power: float
    power: float


if __name__ == '__main__':
    power = 0.

    t_indoor = 21.
    t_building_mass = 20.
    cop = 3

    env_df, _ = ds.get_train_data()
    time = np.arange(len(env_df['time']))

    t_outdoor = np.array(env_df['temperature'])
    temp = np.zeros(len(time))
    bm_temp = np.zeros(len(time))

    for t in time:
        temp[t] = t_indoor
        bm_temp[t] = t_building_mass

        t_indoor, t_building_mass = temperature_simulation(t_outdoor[t], t_indoor, t_building_mass, power, cop)

    plt.plot(time, t_outdoor)
    plt.plot(time, temp, bm_temp)
    plt.show()
