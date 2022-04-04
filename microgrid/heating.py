# Python Libraries
from __future__ import annotations
from typing import List, Tuple, TypeVar
from abc import abstractmethod
from dataclasses import dataclass
import tensorflow as tf

# Local modules
from config import TIME_SLOT
from environment import env
from electrical_asset import ElectricalAsset


"""
    Simulator parameters
"""

Ci = 2.44e6 * 5
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
def temperature_simulation(t_out: T, t_in: float, t_bm: float,
                           hp_power: T, hp_cop: float,
                           solar_rad: float = 0) -> Tuple[T, T]:
    dT_in = 1 / Ci * (
            1 / Ri * (t_bm - t_in) +
            1 / Rvent * (t_out - t_in) +
            (1 - f_rad) * hp_power * hp_cop
    )

    dT_m = 1 / Cm * (
            1 / Ri * (t_in - t_bm) +
            1 / Re * (t_out - t_bm) + gA * solar_rad +
            f_rad * hp_power * hp_cop
    )

    t_in_new = t_in + dT_in * 60 * TIME_SLOT
    t_m_new = t_bm + dT_m * 60 * TIME_SLOT

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

    TEMPERATURE_MARGIN = 1

    def __init__(self, hp: HeatPump, temperature_setpoint: float):
        self.temperature_choice = (temperature_setpoint - self.TEMPERATURE_MARGIN,
                                   temperature_setpoint + self.TEMPERATURE_MARGIN)
        self.hp = hp
        self._time = 0
        self._history: List[float] = []

        self._temperature_setpoint = temperature_setpoint
        self._t_building_mass: float = temperature_setpoint
        self._t_indoor = temperature_setpoint

    @property
    def lower_bound(self) -> float:
        return self.temperature_choice[0]

    @property
    def upper_bound(self) -> float:
        return self.temperature_choice[1]

    @property
    def temperature(self) -> tf.Tensor:
        return tf.convert_to_tensor([self._t_indoor])

    @property
    def power(self) -> tf.Tensor:
        return tf.convert_to_tensor([self.hp.power * self.hp.max_power])

    def get_temperature(self) -> Tuple[tf.Tensor, tf.Tensor]:
        t_out = env.temperature
        solar_rad = 0.      # env.get_irradiation(self._time)

        return temperature_simulation(t_out, self._t_indoor, self._t_building_mass,
                                      self.power, self.hp.cop,
                                      solar_rad)

    def has_heater(self) -> bool:
        return True

    def set_power(self, power: float) -> None:
        self.hp.power = power
        pass

    def step(self) -> None:
        self._history.append(self._t_indoor)
        self._t_indoor, self._t_building_mass = self.get_temperature()
        self._t_indoor = float(self._t_indoor[0])
        self._time += 1

    def reset(self) -> None:
        self._time = 0
        self._history = []
        self._t_indoor = self._temperature_setpoint
        self._t_building_mass = self._temperature_setpoint

    def get_history(self) -> List[float]:
        return self._history


@dataclass
class HeatPump:

    cop: float
    max_power: float
    power: float
