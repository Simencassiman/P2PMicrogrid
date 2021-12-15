# Python Libraries
from typing import List

class Environment:

    def __init__(self):
        self._initialized = False
        self._temperature: List[float] = []
        self._cloud_cover: List[float] = []
        self._humidity: List[float] = []
        self._irradiation: List[float] = []

    def setup(self, temp: List[float], clouds: List[float], humidity: List[float], solar: List[float]) -> None:
        self._temperature = temp
        self._cloud_cover = clouds
        self._humidity = humidity
        self._irradiation = solar
        self._initialized = True

    def get_temperature(self, idx: int) -> float:
        if not self._initialized or idx >= len(self._temperature):
            return 0

        return self._temperature[idx]

    def get_cloud_cover(self, idx: int) -> float:
        if not self._initialized or idx >= len(self._cloud_cover):
            return 0

        return self._cloud_cover[idx]

    def get_humidity(self, idx: int) -> float:
        if not self._initialized or idx >= len(self._humidity):
            return 0

        return self._humidity[idx]

    def get_irradiation(self, idx: int) -> float:
        if not self._initialized or idx >= len(self._irradiation):
            return 0

        return self._irradiation[idx]