from abc import ABC, abstractmethod
import numpy as np

class Compressor(ABC):
    @abstractmethod
    def encode(self, array: np.ndarray) -> bytes:
        pass
    @abstractmethod
    def decode(self, array: bytes) -> np.ndarray:
        pass