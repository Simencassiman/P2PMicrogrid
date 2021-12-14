from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Storage(ABC):

    @abstractmethod
    def charge(self, amount: float) -> None:
        ...

    @abstractmethod
    def discharge(self, amount: float) -> None:
        ...

    @abstractmethod
    def is_full(self) -> bool:
        ...

    @abstractmethod
    def free_space(self) -> float:
        ...


class BatteryStorage(Storage):

    def __init__(self, battery: Battery):
        self.battery = battery
        self._history: List[float] = []

    def charge(self, amount: float) -> None:
        self._history.append(self.battery.soc)
        self.battery.soc += amount

    def discharge(self, amount: float) -> None:
        self._history.append(self.battery.soc)
        self.battery.soc -= amount

    def is_full(self) -> bool:
        return self.battery.soc >= self.battery.max_soc

    def free_space(self) -> float:
        return max(0.0, self.battery.soc - self.battery.min_soc) * self.battery.capacity


class NoStorage(Storage):

    def charge(self, amount: float) -> None:
        pass

    def discharge(self, amount: float) -> None:
        pass

    def is_full(self) -> bool:
        return True

    def free_space(self) -> float:
        return 0


@dataclass
class Battery:

    capacity: float
    peak_power: float
    min_soc: int
    max_soc: int
    efficiency: float
    soc: float
