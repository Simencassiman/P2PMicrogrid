# Python Libraries
from typing import List
from abc import ABC, abstractmethod


class ElectricalAsset(ABC):

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get_history(self) -> List[float]: ...
