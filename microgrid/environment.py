# Python Libraries
from typing import Generator
import tensorflow as tf


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Environment(metaclass=Singleton):

    def __init__(self):
        self._initialized = False
        self._running = False

        self._length: int = 0
        self._time: float = 0.
        self._temperature: float = 0.
        self._dataset: tf.data.Dataset = None

    def setup(self, data: tf.data.Dataset) -> None:
        self._dataset = data
        self._length = len(data)
        self._initialized = True

    @property
    def data(self) -> Generator[tf.Tensor, None, None]:
        if not self._initialized:
            return None

        self._running = True

        for d in self._dataset:
            self._time = d[0][0]
            self._temperature = d[0][1]

            yield d

        self._running = False
        return None

    @property
    def time(self) -> float:
        if not self._running:
            return 0.

        return self._time

    @property
    def temperature(self) -> float:
        if not self._running:
            return 0.

        return self._temperature

    def __len__(self) -> int:
        return self._length


env = Environment()
