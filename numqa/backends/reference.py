"""Golden Reference Oracle Backend running with 64-bit precision."""
import numpy as np
from numqa.backends.base import ComputeBackend


class ReferenceGoldenBackend(ComputeBackend):
    def __init__(self):
        super().__init__(name="GoldenReference_FP64")

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_64 = np.asarray(a, dtype=np.float64)
        b_64 = np.asarray(b, dtype=np.float64)
        return np.matmul(a_64, b_64)

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_64 = np.asarray(x, dtype=np.float64)
        x_max = np.max(x_64, axis=axis, keepdims=True)
        exp_x = np.exp(x_64 - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def cholesky(self, a: np.ndarray) -> np.ndarray:
        a_64 = np.asarray(a, dtype=np.float64)
        return np.linalg.cholesky(a_64)
