# Python Libraries
from __future__ import annotations
from typing import Generator, Tuple, List
from dataclasses import dataclass
from abc import abstractmethod
import tensorflow as tf

# Local modules
from electrical_asset import ElectricalAsset


class Production(ElectricalAsset):

    @property
    @abstractmethod
    def production(self) -> Tuple[tf.Tensor, tf.Tensor]: ...

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...


class Prosumer(Production):

    def __init__(self, pv: PV):
        self.pv = pv
        self._time = 0
        self._production = (p for p in self.pv.production)

    @property
    def production(self) -> Generator[tf.Tensor, None, None]:
        return next(self._production)

    def step(self) -> None:
        self._time += 1

    def get_history(self) -> List[float]:
        return [float(p) for p, _ in self.pv.production]

    def reset(self) -> None:
        self._production = (p for p in self.pv.production)


class Consumer(Production):

    def __init__(self) -> None:
        self._production = (tf.constant([0.]), tf.constant([0.]))

    @property
    def production(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self._production

    def step(self) -> None: ...

    def get_history(self) -> List[float]:
        return []

    def reset(self) -> None: ...


@dataclass
class PV:
    peak_power: float
    production: tf.data.Dataset

