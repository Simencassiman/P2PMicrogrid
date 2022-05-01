# Python Libraries
import re
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

# ------- Parameter setup ------- #
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


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

    # @property
    # def load(self) -> Generator[tf.Tensor, None, None]:
    #     return (l for l in self._load)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]: ...

    def step(self) -> None:
        super(ActingAgent, self).step()
        self.pv.step()
        self.storage.step()
        self.heating.step()

    def reset(self) -> None:
        super(ActingAgent, self).reset()
        self.load = (l for l in self._load)
        self.pv.reset()
        self.storage.reset()
        self.heating.reset()

    def set_profiles(self, load: tf.data.Dataset, pv_gen: tf.data.Dataset) -> None:
        self._load = load
        self.pv.pv.production = pv_gen
        self.reset()


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

    def __init__(self, actor, *args, **kwargs):
        super(RLAgent, self).__init__(*args, **kwargs)

        self.actor = actor
        self.trainer = rl.Trainer(self.actor,
                                  buffer_size=5 * 1000, batch_size=32,
                                  gamma=0.95, tau=0.005,
                                  optimizer=tf.optimizers.Adam(learning_rate=1e-5)
        )

        self._next_load: Tuple[tf.Tensor, tf.Tensor] = next(self.load)
        self._next_production: Tuple[tf.Tensor, tf.Tensor] = self.pv.production

        self._current_balance: tf.Tensor = None
        self._next_balance: tf.Tensor = None
        self._current_state: tf.Tensor = None
        self._action: tf.Tensor = None
        self._price = tf.constant([0])

    def _get_balance(self) -> Tuple[tf.Tensor, tf.Tensor]:
        load, pv = self._next_load, self._next_production

        return tf.expand_dims(load[0] - pv[0], axis=0) / self.max_in,\
               tf.expand_dims(load[1] - pv[1], axis=0) / self.max_in

    def _get_observation_state(self, state: tf.Tensor, balance: tf.Tensor, p2p: tf.Tensor) -> tf.Tensor:
        observation = tf.expand_dims(tf.concat([state[:, 0],
                                                self.heating.normalized_temperature,
                                                balance,
                                                tf.expand_dims(p2p, axis=0)], axis=0), axis=0)

        return observation

    def _divide_power(self, out: tf.Tensor, powers: tf.Tensor) -> tf.Tensor:
        filtered = tf.where(tf.math.sign(out) != tf.math.sign(powers), powers, 0.)
        total_p = tf.math.abs(tf.math.reduce_sum(filtered))

        if total_p == tf.constant([0.]):
            p_out = out * tf.ones(shape=powers.shape) / powers.shape[0]
        else:
            p_out = out * tf.math.abs(filtered) / total_p

        return p_out

    @abstractmethod
    def _act(self) -> tf.Tensor: ...

    def __call__(self, state: tf.Tensor, powers: tf.Tensor,
                 *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:

        p2p = tf.math.reduce_mean(powers) / self.max_in
        self._current_balance, self._next_balance = self._get_balance()

        self._current_state = self._get_observation_state(state, self._current_balance, p2p)

        q_val = self._act()

        p_out = self._divide_power(self._current_balance * self.max_in + self.heating.power,
                                   powers)

        return p_out, q_val

    @abstractmethod
    def take_decision(self, state: tf.Tensor, powers: tf.Tensor,
                      *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]: ...

    @abstractmethod
    def save_memory(self, reward: tf.Tensor, next_state: tf.Tensor, powers: tf.Tensor) -> None: ...

    @abstractmethod
    def train(self, reward: tf.Tensor, next_state: tf.Tensor, powers: tf.Tensor) -> float: ...

    def get_reward(self, cost: tf.Tensor) -> tf.Tensor:
        t_penalty = tf.math.maximum(tf.math.maximum(0., self.heating.lower_bound - self.heating.temperature),
                                    tf.math.maximum(0., self.heating.temperature - self.heating.upper_bound))
        t_penalty = tf.where(t_penalty > 0, t_penalty + 1, 0)

        r = - (cost + 10 * t_penalty)

        return r

    def step(self) -> None:
        super(RLAgent, self).step()
        try:
            self._next_load = next(self.load)
            self._next_production = self.pv.production
        except StopIteration:
            self._next_load = None
            self._next_production = None

    def reset(self) -> None:
        super(RLAgent, self).reset()
        self._next_load = next(self.load)
        self._next_production = self.pv.production

    def load_from_file(self, setting: str, implementation: str) -> None:
        self.actor.load_from_file(f'{re.sub("-", "_", setting)}_{self.id}', implementation)

    def save_to_file(self, setting: str, implementation: str) -> None:
        self.actor.save_to_file(f'{re.sub("-", "_", setting)}_{self.id}', implementation)

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass


class QAgent(RLAgent):

    def __init__(self, *args, **kwargs):
        self._state_quantum = 0.1
        self._num_time_states = 20
        self._num_temp_states = 20
        self._num_balance_states = 20
        self._num_p2p_states = 20

        actor = rl.QActor(self._num_time_states, self._num_temp_states, self._num_balance_states,
                          self._num_p2p_states, decay=0.9)

        super(QAgent, self).__init__(actor, *args, **kwargs)

        self._actions = np.array([0., 0.5, 1.])
        self._last_action: int = -1

    def _act(self) -> tf.Tensor:
        self._last_action, q = self.actor.select_action(self._current_state)
        self.heating.set_power(float(self._actions[self._last_action]))

        return q

    def take_decision(self, state: tf.Tensor, powers: tf.Tensor,
                      *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        p2p = tf.math.reduce_mean(powers) / self.max_in
        current_balance = self._get_balance()[0]
        new_state = self._get_observation_state(state, current_balance, p2p)

        action, q = self.actor.greedy_action(new_state)
        self.heating.set_power(float(self._actions[action]))

        p_out = self._divide_power(current_balance * self.max_in + self.heating.power,
                                   powers)

        return p_out, q

    def save_memory(self, reward: tf.Tensor, next_state: tf.Tensor, powers: tf.Tensor) -> None: ...

    def train(self, reward: tf.Tensor, next_state: tf.Tensor, powers: tf.Tensor) -> float:
        p2p = tf.math.reduce_mean(powers) / self.max_in
        ns = self._get_observation_state(next_state, self._next_balance, p2p)
        self.actor.train(self._current_state, self._last_action, reward, ns)

        return 0.


class DQNAgent(RLAgent):

    def __init__(self, *args, **kwargs) -> None:
        super(DQNAgent, self).__init__(rl.ActorModel(1), *args, **kwargs)

        self.trainer = rl.Trainer(
            self.actor,
            buffer_size=5 * 1000, batch_size=32,
            gamma=0.95, tau=0.005,
            optimizer=tf.optimizers.Adam(learning_rate=1e-5)
        )

    def _act(self) -> tf.Tensor:
        self._action, q_val = self.actor.select_action(self._current_state)
        self.heating.set_power(float(self._action[0]))

        return q_val

    def take_decision(self, state: tf.Tensor, powers: tf.Tensor,
                      *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        p2p = tf.math.reduce_mean(powers) / self.max_in
        current_balance = self._get_balance()[0]
        new_state = self._get_observation_state(state, current_balance, p2p)

        action, q = self.actor.greedy_action(new_state)
        self.heating.set_power(float(action[0]))

        p_out = self._divide_power(current_balance * self.max_in + self.heating.power,
                                   powers)

        return p_out, q[:, 0]

    def save_memory(self, reward: tf.Tensor, next_state: tf.Tensor, powers: tf.Tensor) -> None:
        p2p = tf.math.reduce_mean(powers) / self.max_in
        ns = self._get_observation_state(next_state, self._next_balance, p2p)
        self.trainer.buffer.add(self._current_state[0, :], self._action, reward, ns[0, :])

    def train(self, reward: tf.Tensor, next_state: tf.Tensor, powers: tf.Tensor) -> float:
        self.save_memory(reward, next_state, powers)
        loss = self.trainer.train()

        return loss

    def load_from_file(self, setting: str, implementation: str) -> None:
        super(DQNAgent, self).load_from_file(setting, implementation)
        self.trainer.load_from_file(f'{re.sub("-", "_", setting)}_{self.id}', implementation)

    def save_to_file(self, setting: str, implementation: str) -> None:
        super(DQNAgent, self).save_to_file(setting, implementation)
        self.trainer.save_to_file(f'{re.sub("-", "_", setting)}_{self.id}', implementation)
