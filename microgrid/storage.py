# Python Libraries
from __future__ import annotations
from typing import List
from abc import abstractmethod
from dataclasses import dataclass
import numpy as np

# Local modules
from electrical_asset import ElectricalAsset


class Storage(ElectricalAsset):

    @property
    @abstractmethod
    def is_full(self) -> bool:
        ...

    @property
    @abstractmethod
    def available_space(self) -> float:
        ...

    @property
    @abstractmethod
    def available_energy(self) -> float:
        ...

    @abstractmethod
    def to_soc(self, energy: float) -> float:
        ...

    @abstractmethod
    def charge(self, amount: float) -> None:
        ...

    @abstractmethod
    def discharge(self, amount: float) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...


class BatteryStorage(Storage):

    def __init__(self, battery: Battery):
        self.battery = battery
        self._time = 0
        self._history: List[float] = []

    @property
    def is_full(self) -> bool:
        return self.battery.soc >= self.battery.max_soc

    @property
    def available_space(self) -> float:
        return (max(0.0, self.battery.max_soc - self.battery.soc) * self.battery.capacity /
                np.sqrt(self.battery.efficiency))

    @property
    def available_energy(self) -> float:
        return (max(0.0, self.battery.soc - self.battery.min_soc) * self.battery.capacity *
               np.sqrt(self.battery.efficiency))

    def to_soc(self, energy: float) -> float:
        return energy / self.battery.capacity

    def charge(self, amount: float) -> None:
        self.battery.soc += np.sqrt(self.battery.efficiency) * amount

    def discharge(self, amount: float) -> None:
        self.battery.soc -= amount / np.sqrt(self.battery.efficiency)

    def step(self) -> None:
        self._history.append(self.battery.soc)
        self._time += 1

    def reset(self) -> None:
        self._time = 0
        self._history = []
        self.battery.soc = 0.5

    def get_history(self) -> List[float]:
        return self._history


class NoStorage(Storage):

    @property
    def is_full(self) -> bool:
        return True

    @property
    def available_space(self) -> float:
        return 0

    @property
    def available_energy(self) -> float:
        return 0

    def to_soc(self, energy: float) -> float:
        return 0

    def charge(self, amount: float) -> None: ...

    def discharge(self, amount: float) -> None: ...

    def step(self) -> None: ...

    def reset(self) -> None: ...

    def get_history(self) -> List[float]:
        return []


@dataclass
class Battery:

    capacity: float
    peak_power: float
    min_soc: float
    max_soc: float
    efficiency: float
    soc: float
