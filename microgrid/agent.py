# Python Libraries
from typing import List, Tuple, Generator, Optional
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

# Local modules
from config import TIME_SLOT, MINUTES_PER_HOUR, HOURS_PER_DAY, CENTS_PER_EURO
import config as cf
from environment import env
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
    def take_decision(self, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
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

        self._cost_avg = cf.GRID_COST_AVG
        self._cost_amplitude = cf.GRID_COST_AMPLITUDE
        self._cost_phase = cf.GRID_COST_PHASE
        self._cost_frequency = 2 * np.pi * MINUTES_PER_HOUR / TIME_SLOT * HOURS_PER_DAY / cf.GRID_COST_PERIOD
        self._cost_normalization = CENTS_PER_EURO

        self._injection_price = tf.convert_to_tensor([cf.GRID_INJECTION_PRICE])

    def take_decision(self, state: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        cost = (
                (self._cost_avg
                 + self._cost_amplitude
                 * tf.math.sin(state[0] * self._cost_frequency + self._cost_phase)
                 ) / self._cost_normalization  # from c€ to €
        )

        return cost, self._injection_price

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass


class ActingAgent(Agent, ABC):

    def __init__(self, load: tf.data.Dataset, production: Production, storage: Storage, heating: Heating,
                 max_in: float, max_out: float, *args, **kwargs):
        super(ActingAgent, self).__init__()
        self.max_in = max_in
        self.max_out = max_out

        self._load = load
        self.load: Generator[tf.Tensor, None, None] = (l for l in load)
        self.pv = production
        self.storage = storage
        self.heating = heating


class RuleAgent(ActingAgent):

    def __init__(self, *args, **kwargs):
        super(RuleAgent, self).__init__(*args, **kwargs)

    def reset(self) -> None:
        super(RuleAgent, self).reset()
        self.load = (l for l in self._load)
        self.pv.reset()
        self.storage.reset()
        self.heating.reset()

    def take_decision(self, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:

        # Check temperature
        self._update_heating()

        # Combine load with solar generation
        current_load, _ = next(self.load)
        current_pv, _ = self.pv.production
        current_balance = current_load + current_pv

        # Combine balance with heating
        current_power = current_balance + self.heating.power

        # Step through all times
        self._step()

        # return resulting in/output to grid
        return current_power, tf.constant([0.])

    def _update_heating(self) -> None:
        temperature = self.heating.temperature

        if temperature[0] <= self.heating.lower_bound:
            self.heating.set_power(1)
        elif temperature[0] >= self.heating.upper_bound:
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

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass

    def _step(self) -> None:
        self.pv.step()
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
