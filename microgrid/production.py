from __future__ import  annotations

from dataclasses import dataclass
from typing import List
from abc import ABC, abstractmethod


class Production(ABC):

    @abstractmethod
    def produce(self, time: int) -> float:
        ...


class Prosumer(Production):

    def __init__(self, pv: PV):
        self.pv = pv

    def produce(self, time: int) -> float:
        return self.pv.production[time] if time in range(len(self.pv.production)) else 0


class Consumer(Production):

    def produce(self, time: int) -> float:
        return 0


@dataclass
class PV:
    peak_power: float
    production: List[float]

