from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


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
        self._history = []
        self._indoor_temperature = temperature_setpoint

    def get_temperature(self) -> float:
        pass

    def has_heater(self) -> bool:
        pass

    def set_power(self, power: float) -> None:
        pass


@dataclass
class HeatPump:

    cop: float
    max_power: float
    power: float
