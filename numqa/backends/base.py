"""Abstract Base Class for all compute backends."""
from abc import ABC, abstractmethod
import numpy as np


class ComputeBackend(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """General Matrix Multiplication: C = A @ B"""
        pass

    @abstractmethod
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax activation function"""
        pass

    @abstractmethod
    def cholesky(self, a: np.ndarray) -> np.ndarray:
        """Lower-triangular Cholesky factor L: A = L @ L.T"""
        pass
