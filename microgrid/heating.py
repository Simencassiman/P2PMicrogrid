# Python Libraries
from __future__ import annotations
from typing import List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Local modules
from config import TIME_SLOT
from microgrid import environment


"""
    Simulator parameters
"""

Ci = 2.44e6
Cm = 9.4e7
Ri = 8.64e-4
Re = 1.05e-2
Rvent = 7.98e-3
gA = 11.468
f_rad = 0.3

"""
    End parameters
"""


class Heating(ABC):

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
    def temperature(self) -> float:
        ...

    @property
    @abstractmethod
    def power(self) -> float:
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
        self._t_building_mass: float = temperature_setpoint
        self._t_indoor = temperature_setpoint

    @property
    def lower_bound(self) -> float:
        return self.temperature_choice[0]

    @property
    def upper_bound(self) -> float:
        return self.temperature_choice[1]

    @property
    def temperature(self) -> float:
        return self._t_indoor

    @property
    def power(self) -> float:
        return self.hp.power * self.hp.max_power

    def get_temperature(self) -> Tuple[float, float]:
        t_out = environment.get_temperature(self._time)
        solar_rad = environment.get_irradiation(self._time)

        dT_in = 1/Ci * (
            1/Ri * (self._t_building_mass - self._t_building_mass) +
            1/Rvent * (t_out - self._t_indoor) +
            (1 - f_rad) * self.power * self.hp.cop
        )

        dT_m = 1/Cm * (
            1/Ri * (self._t_indoor - self._t_building_mass) +
            1/Re * (t_out - self._t_building_mass) + gA * solar_rad +
            f_rad * self.power * self.hp.cop
        )

        t_in_new = self._t_indoor + dT_in * 60 * TIME_SLOT
        t_m_new = self._t_building_mass + dT_m * 60 * TIME_SLOT

        return t_in_new, t_m_new

    def has_heater(self) -> bool:
        return True

    def set_power(self, power: float) -> None:
        self.hp.power = power
        pass

    def step(self) -> None:
        self._history.append(self._t_indoor)
        self._t_indoor, self._t_building_mass = self.get_temperature()
        self._time += 1


@dataclass
class HeatPump:

    cop: float
    max_power: float
    power: float
