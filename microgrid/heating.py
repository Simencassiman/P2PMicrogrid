from __future__ import annotations
from typing import List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


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

    @abstractmethod
    def get_temperature(self) -> float:
        ...

    @abstractmethod
    def has_heater(self) -> bool:
        ...

    @abstractmethod
    def set_power(self, power: float) -> None:
        ...


class HPHeating(Heating):

    TEMPERATURE_MARGIN = 1

    def __init__(self, hp: HeatPump, temperature_setpoint: float):
        self.temperature_choice = (temperature_setpoint - self.TEMPERATURE_MARGIN,
                                   temperature_setpoint + self.TEMPERATURE_MARGIN)
        self.hp = hp
        self._history: List[float] = []
        self._t_building_mass: float = 0
        self._t_indoor = temperature_setpoint
        self._power: float = 0

    @property
    def temperature(self) -> float:
        return self._t_indoor

    @property
    def power(self) -> float:
        return self._power

    def get_temperature(self) -> Tuple[float, float]:
        t_out = 0       # environment.get_temperature()
        solar_rad = 0   # environment.get_irradiation()

        t_in = 1/Ci * (
            1/Ri * (self._t_building_mass - self._t_building_mass) +
            1/Rvent * (t_out - self._t_indoor) +
            (1 - f_rad) * self._power * self.hp.cop
        )

        t_m = 1/Cm * (
            1/Ri * (self._t_indoor - self._t_building_mass) +
            1/Re * (t_out - self._t_building_mass) + gA * solar_rad +
            f_rad * self._power * self.hp.cop
        )
        return t_in, t_m

    def has_heater(self) -> bool:
        return True

    def set_power(self, power: float) -> None:
        self._t_indoor, self._t_building_mass = self.get_temperature()
        self._history.append(self._t_indoor)
        self._power = power


@dataclass
class HeatPump:

    cop: float
    max_power: float
    power: float
