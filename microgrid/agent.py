# Python Libraries
from typing import List, Tuple, Generator, Optional
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

# Local modules
from config import TIME_SLOT, MINUTES_PER_HOUR, HOURS_PER_DAY, CENTS_PER_EURO
import config as cf
from controller import BackupController, BuildingController, ChargingController
from production import Production
from storage import Storage
from heating import Heating
import rl


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

    def step(self) -> None:
        self.time += 1

    def reset(self) -> None:
        self.time = 0


class GridAgent(Agent):

    def __init__(self):
        super(GridAgent, self).__init__()

        self._cost_avg = cf.GRID_COST_AVG
        self._cost_amplitude = cf.GRID_COST_AMPLITUDE
        self._cost_phase = cf.GRID_COST_PHASE
        self._cost_frequency = 2 * np.pi * HOURS_PER_DAY / cf.GRID_COST_PERIOD
        self._cost_normalization = CENTS_PER_EURO

        self._injection_price = tf.convert_to_tensor([cf.GRID_INJECTION_PRICE])

    def take_decision(self, state: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        cost = (
                (self._cost_avg
                 + self._cost_amplitude
                 * tf.math.sin(state[0] * self._cost_frequency - self._cost_phase)
                 ) / self._cost_normalization  # from c€ to €
        )

        return tf.cast(cost, dtype=tf.float32), self._injection_price

    def _predict(self) -> List: ...

    def _communicate(self) -> List: ...


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

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]: ...

    def step(self) -> None:
        super(ActingAgent, self).step()
        self.pv.step()
        self.storage.step()
        self.heating.step()

    def reset(self) -> None:
        super(ActingAgent, self).reset()
        self.load = (l for l in self._load.as_numpy_iterator())
        self.pv.reset()
        self.storage.reset()
        self.heating.reset()


class RuleAgent(ActingAgent):

    def __init__(self, *args, **kwargs):
        super(RuleAgent, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.take_decision(*args, **kwargs)

    def take_decision(self, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:

        # Check temperature
        self._update_heating()

        # Combine load with solar generation
        current_load, _ = next(self.load)
        current_pv, _ = self.pv.production
        current_balance = current_load - current_pv

        # Combine balance with heating
        current_power = current_balance + self.heating.power

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


class RLAgent(ActingAgent):

    def __init__(self, *args, **kwargs):
        super(RLAgent, self).__init__(*args, **kwargs)
        # self.backup = controller

        self.actor = rl.ActorModel(1)
        self.trainer = rl.Trainer(self.actor,
                                  buffer_size=30 * 1000, batch_size=32,
                                  gamma=0.95, tau=0.005,
                                  optimizer=tf.optimizers.Adam(learning_rate=1e-5)
        )

        self._current_balance: tf.Tensor = None
        self._next_balance: tf.Tensor = None
        self._current_state: tf.Tensor = None
        self._action: tf.Tensor = None
        self._price = tf.constant([0])

    def _get_balance(self) -> Tuple[tf.Tensor, tf.Tensor]:
        load, pv = next(self.load), self.pv.production

        return tf.expand_dims(load[0] - pv[0], axis=0) / self.max_in,\
               tf.expand_dims(load[1] - pv[1], axis=0) / self.max_in

    def _get_observation_state(self, state: tf.Tensor, balance: tf.Tensor) -> tf.Tensor:
        observation = tf.expand_dims(tf.concat([state[:, 0],
                                                self.heating.normalized_temperature,
                                                balance], axis=0), axis=0)

        return observation

    def __call__(self, state: tf.Tensor, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        self._current_balance, self._next_balance = self._get_balance()

        self._current_state = self._get_observation_state(state, self._current_balance)

        self._action, q_val = self.actor(self._current_state)
        self.heating.set_power(float(self._action[0]))

        p_out = self._current_balance * self.max_in + self.heating.power
        return p_out, self._price, q_val

    def explore(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        self._current_balance, self._next_balance = self._get_balance()
        self._current_state = self._get_observation_state(state, self._current_balance)

        self._action, q = self.actor.random_action()

        self.heating.set_power(float(self._action[0]))

        return self._current_balance * self.max_in + self.heating.power, self._price, q

    def take_decision(self, state: tf.Tensor, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        current_balance = self._get_balance()[0]
        new_state = self._get_observation_state(state, current_balance)

        action, q = self.actor.greedy_action(new_state)
        self.heating.set_power(float(action[0]))

        return current_balance * self.max_in + self.heating.power, self._price, q[:, 0]

    def train(self, reward: tf.Tensor, next_state: tf.Tensor) -> float:
        self.save_memory(reward, next_state)
        loss = self.trainer.train()

        return loss

    def get_reward(self, action: tf.Tensor, c_buy: tf.Tensor, c_inj: tf.Tensor) -> tf.Tensor:
        cost = tf.where(tf.math.greater(action, 0), action * c_buy, action * c_inj)

        t_penalty = tf.math.maximum(tf.math.maximum(0., self.heating.lower_bound - self.heating.temperature),
                                    tf.math.maximum(0., self.heating.temperature - self.heating.upper_bound))
        t_penalty = tf.where(t_penalty > 0, t_penalty + 1, 0)

        r = - (cost + 10 * t_penalty)

        return r

    def save_memory(self, reward: tf.Tensor, next_state: tf.Tensor) -> None:
        ns = self._get_observation_state(next_state, self._next_balance)
        self.trainer.buffer.add(self._current_state[0, :], self._action, reward, ns[0, :])

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass


class QAgent(RLAgent):

    def __init__(self, *args, **kwargs):
        super(QAgent, self).__init__(*args, **kwargs)

        self._state_quantum = 0.1
        self._num_time_states = 20
        self._num_temp_states = 20
        self._num_balance_states = 20

        self._actions = np.array([0., 0.5, 1.])
        self.actor = rl.QActor(self._num_time_states, self._num_temp_states, self._num_balance_states)

        self._last_action: int = -1

    def __call__(self, state: tf.Tensor, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        self._current_balance, self._next_balance = self._get_balance()
        self._current_state = self._get_observation_state(state, self._current_balance)

        self._last_action, q = self.actor.select_action(self._current_state)
        self.heating.set_power(self._actions[self._last_action])

        p_out = self._current_balance * self.max_in + self.heating.power
        return p_out, self._price, q

    def take_decision(self, state: tf.Tensor, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        current_balance = self._get_balance()[0]
        new_state = self._get_observation_state(state, current_balance)

        action, q = self.actor.greedy_action(new_state)
        self.heating.set_power(float(action))

        return current_balance * self.max_in + self.heating.power, self._price, q

    def train(self, reward: tf.Tensor, next_state: tf.Tensor) -> float:
        ns = self._get_observation_state(next_state, self._next_balance)
        self.actor.train(self._current_state, self._last_action, reward, ns)

        return 0.

    def save_memory(self, reward: tf.Tensor, next_state: tf.Tensor) -> None: ...


