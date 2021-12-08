# Python Libraries
from typing import List

# Local Modules
from .agent import ActingAgent, GridAgent


class CommunityMicrogrid:

    def __init__(self):
        self.agents: List[ActingAgent]
        self.grid: GridAgent

    def run(self) -> List:
        pass

    def train(self) -> None:
        pass

    def _step(self) -> None:
        pass

    def _iterate(self) -> None:
        pass
