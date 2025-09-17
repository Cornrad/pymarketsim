from abc import ABC, abstractmethod
from typing import List

from ..fourheap.order import Order


class Agent(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def take_action(self) -> List[Order]:
        pass

    @abstractmethod
    def reset(self):
        pass

    def get_pos_value(self) -> float:
        pass

