from __future__ import annotations
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
        self._history = []

    def charge(self, amount: float) -> None:
        pass

    def discharge(self, amount: float) -> None:
        pass

    def is_full(self) -> bool:
        pass

    def free_space(self) -> float:
        pass


class NoStorage(Storage):

    def charge(self, amount: float) -> None:
        pass

    def discharge(self, amount: float) -> None:
        pass

    def is_full(self) -> bool:
        pass

    def free_space(self) -> float:
        pass


@dataclass
class Battery:

    capacity: float
    peak_power: float
    min_soc: int
    max_soc: int
    efficiency: float
    soc: float
