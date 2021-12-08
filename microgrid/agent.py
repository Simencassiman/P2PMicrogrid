# Python Libraries
from typing import List, Optional
from abc import ABC, abstractmethod

# Local modules
from .controller import BackupController, BuildingController, ChargingController
from .production import Production
from .storage import Storage
from .heating import Heating
from .ev import EV


class Agent(ABC):

    def __init__(self):
        self.id: int
        self.time: int = 0

    @abstractmethod
    def take_decision(self, **kwargs) -> List:
        ...

    @abstractmethod
    def _predict(self) -> List:
        ...

    @abstractmethod
    def _communicate(self) -> List:
        ...


class GridAgent(Agent):

    def __int__(self, prices: Optional[List[float]] = None):
        super(GridAgent, self).__init__()
        self.prices = prices

    def take_decision(self, **kwargs) -> List:
        pass

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass


class ActingAgent(Agent, ABC):

    def __init__(self, max_in: float, max_out: float):
        super(ActingAgent, self).__init__()
        self.max_in = max_in
        self.max_out = max_out


class RuleAgent(ActingAgent):

    def __init__(self, load: List[float], production: Production, storage: Storage, heating: Heating,
                 *args, **kwargs):
        super(RuleAgent, self).__init__(*args, **kwargs)
        self.load = load
        self.production = production
        self.storage = storage
        self.heating = heating

    def take_decision(self, **kwargs) -> List:
        pass

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass


# TODO: inherit from tensorflow model
class RLAgent(ActingAgent, ABC):

    def __init__(self, controller: BackupController, *args, **kwargs):
        super(RLAgent, self).__init__(*args, **kwargs)
        self.backup = controller
        self._layers: List


class PrivateAgent(RLAgent, ABC):

    def __init__(self, load: List[float], *args, **kwargs):
        super(PrivateAgent, self).__init__(*args, **kwargs)
        self._base_load = load


class GatewayAgent(RLAgent):

    def __init__(self, agents: List[ActingAgent], *args, **kwargs):
        super(GatewayAgent, self).__init__(*args, **kwargs)
        self.agents = agents

    def take_decision(self, **kwargs) -> List:
        pass

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass


class BuildingAgent(PrivateAgent):

    def __init__(self, load: List[float], production: Production, storage: Storage, heating: Heating,
                 *args, **kwargs):
        super(BuildingAgent, self).__init__(load, *args, controller=BuildingController(self), **kwargs)
        self.production = production
        self.storage = storage
        self.heating = heating

    def take_decision(self, **kwargs) -> List:
        pass

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass


class ChargingAgent(PrivateAgent):

    def __init__(self, evs: List[EV], *args, **kwargs):
        super(ChargingAgent, self).__init__(*args, load=[], controller=ChargingController(self), **kwargs)
        self.evs = evs

    def take_decision(self, **kwargs) -> List:
        pass

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass
