from dataclasses import dataclass


@dataclass
class EV:

    capacity: float
    max_power: float
    efficiency: float
    soc: float

    def charge(self, amount: float) -> None:
        pass

    def is_full(self) -> bool:
        pass
