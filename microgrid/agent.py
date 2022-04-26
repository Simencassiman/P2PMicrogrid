# Python Libraries
import re
from typing import List, Tuple, Generator
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Local modules
import heating
from config import TIME_SLOT, HOURS_PER_DAY, CENTS_PER_EURO
import config as cf
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
        current_balance = tf.expand_dims(current_load - current_pv, axis=0)

        # Combine balance with heating
        current_power = current_balance + tf.squeeze(self.heating.power)

        # return resulting in/output to grid
        return current_power, tf.constant([0])

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


class BaselineAgent(ActingAgent):

    def __init__(self, *args, **kwargs) -> None:
        super(BaselineAgent, self).__init__(*args, **kwargs)

        self._hp_power = np.array([   0,           0,           0,           0,           0,    0,         204.1183166, 223.3856048, 242.4940185, 262.3419799, 3000,        3000,           0,           0,           0,    0,           0,           0,         321.2681884, 336.4851074,  351.5411071, 368.080383,  384.471527,  400.7154846, 416.8153686,  429.7907409, 442.6157836, 455.2926330, 467.8244018, 481.8542175,  495.749481,  509.5132751, 523.1467895, 518.6010131, 513.8638305,  508.9374389, 503.8260803, 478.0937,    452.1070861, 734.245666, 1256.591674, 1599.0556640, 353.7709350, 330.6316223,2471.9882812,  290.1255493, 272.7893066, 255.2711639, 237.5742950,3000,  196.872741,  176.2528686, 155.4573669, 141.9486694, 128.2969970, 3000,         100.5762329,  92.0312118,   0,        3000,    0,           0,           0,           0,           0,  992.1799926,   0,           0,         322.2501220,  75.5588378,  184.4559326, 118.2068,    138.3226928, 157.3858032, 176.4374542,  195.4774322, 214.5069274, 225.4690399, 236.3911743, 247.273437,  258.116668,  267.2800903, 276.4003906, 285.477600,  294.5127868,  302.6107788, 310.6646118, 318.67535,   326.6430053, 332.480041,  338.267700,  344.0069885, 349.6990051, 346.3937377, 343.0097351,  339.54888916])

    def __call__(self, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        time = round(float(args[0][0, 0]) * 96)
        power = tf.expand_dims(next(self.load)[0] - self.pv.production[0], axis=0)

        # if time < 60:
        #     hp_power = self._hp_power[time]
        # elif self.heating.temperature <= 20:
        #     req_power = heating.required_power(self.heating.temperature, self.heating._t_building_mass,
        #                                        self.heating.hp.cop)
        #     hp_power = tf.math.minimum(req_power[0], self.heating.hp.max_power)
        # else:
        #     hp_power = 0

        hp_power = self._hp_power[time]

        # if self.heating.temperature <= 20.5 and power < 0:
        #     hp_power = tf.math.minimum(tf.math.maximum(tf.math.abs(power), hp_power), self.heating.hp.max_power)

        power += hp_power
        self.heating.set_power(float(hp_power) / self.heating.hp.max_power)

        # if self.heating.temperature <= 20:
        #     req_power = heating.required_power(self.heating.temperature, self.heating._t_building_mass, self.heating.hp.cop)
        #     hp_power = tf.math.minimum(req_power[0], self.heating.hp.max_power)
        #     power += hp_power
        #     self.heating.set_power(float(hp_power / self.heating.hp.max_power))
        # else:
        #     self.heating.set_power(0.)

        return power, tf.constant([0])

    def take_decision(self, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        return self(*args, **kwargs)

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass


class RLAgent(ActingAgent):

    def __init__(self, *args, **kwargs):
        super(RLAgent, self).__init__(*args, **kwargs)
        # self.backup = controller

        self.actor = rl.ActorModel(epsilon=0.4782969)
        self.trainer = rl.Trainer(self.actor,
                                  buffer_size=20 * 1000, batch_size=32,
                                  gamma=0.95, tau=0.005,
                                  optimizer=tf.optimizers.Adam(learning_rate=1e-5)
        )

        self._next_load: Tuple[tf.Tensor, tf.Tensor] = next(self.load)
        self._next_production: Tuple[tf.Tensor, tf.Tensor] = self.pv.production

        self._current_balance: tf.Tensor = None
        self._next_balance: tf.Tensor = None
        self._current_state: tf.Tensor = None
        self._action: tf.Tensor = None

    def _get_balance(self) -> Tuple[tf.Tensor, tf.Tensor]:
        load, pv = self._next_load, self._next_production

        return tf.expand_dims(load[0] - pv[0], axis=0) / self.max_in,\
               tf.expand_dims(load[1] - pv[1], axis=0) / self.max_in

    def _get_observation_state(self, state: tf.Tensor, balance: tf.Tensor) -> tf.Tensor:
        observation = tf.expand_dims(tf.concat([state[:, 0],
                                                self.heating.normalized_temperature,
                                                balance], axis=0), axis=0)

        return observation

    def __call__(self, state: tf.Tensor, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        self._current_balance, self._next_balance = self._get_balance()

        self._current_state = self._get_observation_state(state, self._current_balance)

        self._action, q_val = self.actor(self._current_state)
        self.heating.set_power(float(self._action[0]))

        p_out = self._current_balance * self.max_in + self.heating.power

        return p_out, q_val

    def explore(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        self._current_balance, self._next_balance = self._get_balance()
        self._current_state = self._get_observation_state(state, self._current_balance)

        self._action, q = self.actor.random_action()

        self.heating.set_power(float(self._action[0]))

        p_out = self._current_balance * self.max_in + self.heating.power

        return p_out, q

    def take_decision(self, state: tf.Tensor, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        current_balance = self._get_balance()[0]
        new_state = self._get_observation_state(state, current_balance)

        action, q = self.actor.greedy_action(new_state)
        self.heating.set_power(float(action[0]))

        p_out = current_balance * self.max_in + self.heating.power

        return p_out, q[:, 0]

    def train(self, reward: tf.Tensor, next_state: tf.Tensor) -> float:
        self.save_memory(reward, next_state)
        loss = self.trainer.train()

        return loss

    def get_reward(self, cost: tf.Tensor) -> tf.Tensor:
        t_penalty = tf.math.maximum(tf.math.maximum(0., self.heating.lower_bound - self.heating.temperature),
                                    tf.math.maximum(0., self.heating.temperature - self.heating.upper_bound))
        t_penalty = tf.where(t_penalty > 0, t_penalty + 1, 0)

        r = - (cost + 10 * t_penalty)

        return r

    def save_memory(self, reward: tf.Tensor, next_state: tf.Tensor) -> None:
        ns = self._get_observation_state(next_state, self._next_balance)
        self.trainer.buffer.add(self._current_state[0, :], self._action, reward, ns[0, :])

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

    def _predict(self) -> List:
        pass

    def _communicate(self) -> List:
        pass

    def load_from_file(self, setting: str, implementation: str) -> None:
        self.actor.load_from_file(f'{re.sub("-", "_", setting)}_{self.id}', implementation)
        self.trainer.load_from_file(f'{re.sub("-", "_", setting)}_{self.id}', implementation)

    def save_to_file(self, setting: str, implementation: str) -> None:
        self.actor.save_to_file(f'{re.sub("-", "_", setting)}_{self.id}', implementation)
        self.trainer.save_to_file(f'{re.sub("-", "_", setting)}_{self.id}', implementation)

class QAgent(RLAgent):

    def __init__(self, *args, **kwargs):
        super(QAgent, self).__init__(*args, **kwargs)

        self._state_quantum = 0.1
        self._num_time_states = 20
        self._num_temp_states = 20
        self._num_balance_states = 20

        self._actions = np.array([0., 0.5, 1.])
        self.actor = rl.QActor(self._num_time_states, self._num_temp_states, self._num_balance_states, decay=0.9)

        self._last_action: int = -1

    def __call__(self, state: tf.Tensor, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        self._current_balance, self._next_balance = self._get_balance()
        self._current_state = self._get_observation_state(state, self._current_balance)

        self._last_action, q = self.actor.select_action(self._current_state)
        self.heating.set_power(float(self._actions[self._last_action]))

        p_out = self._current_balance * self.max_in + self.heating.power

        return p_out, q

    def take_decision(self, state: tf.Tensor, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        current_balance = self._get_balance()[0]
        new_state = self._get_observation_state(state, current_balance)

        action, q = self.actor.greedy_action(new_state)
        self.heating.set_power(float(self._actions[action]))

        p_out = current_balance * self.max_in + self.heating.power

        return p_out, q

    def train(self, reward: tf.Tensor, next_state: tf.Tensor) -> float:
        ns = self._get_observation_state(next_state, self._next_balance)
        self.actor.train(self._current_state, self._last_action, reward, ns)

        return 0.

    def save_memory(self, reward: tf.Tensor, next_state: tf.Tensor) -> None: ...

    def load_from_file(self, setting: str, implementation: str) -> None:
        self.actor.load_from_file(f'{re.sub("-", "_", setting)}_{self.id}', implementation)

    def save_to_file(self, setting: str, implementation: str) -> None:
        self.actor.save_to_file(f'{re.sub("-", "_", setting)}_{self.id}', implementation)


if __name__ == '__main__':

    cost = (
            (cf.GRID_COST_AVG
             + cf.GRID_COST_AMPLITUDE
             * np.sin(np.arange(96) / 96 * 2 * np.pi * HOURS_PER_DAY / cf.GRID_COST_PERIOD - cf.GRID_COST_PHASE)
             ) / CENTS_PER_EURO # from c€ to €
    )

    print(cost[58:62])
