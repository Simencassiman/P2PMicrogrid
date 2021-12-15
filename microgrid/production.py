from __future__ import  annotations
from dataclasses import dataclass
from typing import List
from abc import ABC, abstractmethod
import numpy.typing as npt


class Production(ABC):

    @abstractmethod
    def produce(self) -> float:
        ...

    @abstractmethod
    def step(self) -> None:
        ...


class Prosumer(Production):

    def __init__(self, pv: PV):
        self.pv = pv
        self._time = 0

    def produce(self) -> float:
        return self.pv.production[self._time] if self._time in range(len(self.pv.production)) else 0

    def step(self) -> None:
        self._time += 1


class Consumer(Production):

    def produce(self) -> float:
        return 0

    def step(self) -> None:
        ...


@dataclass
class PV:
    peak_power: float
    production: npt.ArrayLike[float]

