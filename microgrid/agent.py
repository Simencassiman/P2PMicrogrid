# Python Libraries
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

# Local modules
from config import TIME_SLOT, MINUTES_PER_HOUR, HOURS_PER_DAY, CENTS_PER_EURO
import config as cf
from controller import BackupController, BuildingController, ChargingController
from production import Production
from storage import Storage
from heating import Heating
from ev import EV


class Agent(ABC):

    __last_id = -1

    def __init__(self):
        self.id = self.__last_id + 1
        Agent.__last_id += 1
        self.time: int = 0

    @classmethod
    def reset_ids(cls) -> None:
        cls.__last_id = -1

    @abstractmethod
    def take_decision(self, **kwargs) -> Tuple[float, float]:
        ...

    @abstractmethod
    def _predict(self) -> List:
        ...

    @abstractmethod
    def _communicate(self) -> List:
        ...

    def _step(self) -> None:
        self.time += 1

    def reset(self) -> None:
        self.time = 0


class GridAgent(Agent):

    def __init__(self):
        super(GridAgent, self).__init__()
        self.prices = (
                (cf.GRID_COST_AVG +
                 cf.GRID_COST_AMPLITUDE *
                 np.sin(2 * np.pi * np.array([t / (MINUTES_PER_HOUR / TIME_SLOT)
                                              for t in range(int(MINUTES_PER_HOUR / TIME_SLOT * HOURS_PER_DAY))])
                        / cf.GRID_COST_PERIOD + cf.GRID_COST_PHASE)
                 ) / CENTS_PER_EURO   # from c€ to €
        )
        self.injection_price: float = np.min(self.prices)

    def take_decision(self, **kwargs) -> Tuple[float, float]:
        cost = self.prices[self.time % int(MINUTES_PER_HOUR / TIME_SLOT * HOURS_PER_DAY)]
        self._step()

        return cost, self.injection_price

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

    def __init__(self, load: np.ndarray, production: Production, storage: Storage, heating: Heating,
                 *args, **kwargs):
        super(RuleAgent, self).__init__(*args, **kwargs)
        self.load = load
        self.production = production
        self.storage = storage
        self.heating = heating

    def _update_heating(self) -> None:
        temperature = self.heating.temperature

        if temperature <= self.heating.lower_bound:
            self.heating.set_power(1)
        elif temperature >= self.heating.upper_bound:
            self.heating.set_power(0)

    def _update_storage(self, balance: float) -> float:

        energy = balance * 60 * TIME_SLOT
        if balance > 0 and self.storage.available_energy > 0:
            # If not enough production and battery is not empty
            to_extract = min(energy, self.storage.available_energy)
            self.storage.discharge(self.storage.to_soc(to_extract))
            balance -= to_extract / (60 * TIME_SLOT)

        elif balance < 0 and not self.storage.is_full:
            # If too much production and battery is not full
            to_store = min(-energy, self.storage.available_space)
            self.storage.charge(self.storage.to_soc(to_store))
            balance += to_store / (60 * TIME_SLOT)

        return balance

    def take_decision(self, **kwargs) -> Tuple[float, float]:

        # Check temperature
        self._update_heating()

        # Combine load with heating
        current_power = self.load[self.time] + self.heating.power

        # Check production
        current_production = self.production.produce()
        balance = self._update_storage(current_power - current_production)

        # Step through all times
        self._step()

        # return resulting in/output to grid
        return balance, 0

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass

    def _step(self) -> None:
        self.production.step()
        self.storage.step()
        self.heating.step()
        super(RuleAgent, self)._step()


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
