# Python Libraries
from typing import TYPE_CHECKING
from typing import List
from abc import ABC, abstractmethod

# Local modules
if TYPE_CHECKING:
    from agent import RLAgent, BuildingAgent, ChargingAgent


class BackupController(ABC):

    def __init__(self, agent: """RLAgent"""):
        self.agent = agent

    @abstractmethod
    def shield(self, decision: List) -> List:
        ...


class BuildingController(BackupController):

    def __init__(self, agent: """BuildingAgent"""):
        super(BuildingController, self).__init__(agent)

    def shield(self, decision: List) -> List:
        pass


class ChargingController(BackupController):

    def __init__(self, agent: """ChargingAgent"""):
        super(ChargingController, self).__init__(agent)

    def shield(self, decision: List) -> List:
        pass
