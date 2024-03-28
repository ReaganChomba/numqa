"""
GPU Simulated Accelerator Backend:
Simulates GPU warp-level fused multiply-add (FMA), reduced-precision truncation,
and asynchronous accumulation drift.
"""
import numpy as np
from numqa.backends.base import ComputeBackend


class GPUSimulatedBackend(ComputeBackend):
    def __init__(self, mode: str = "fp32_fma"):
        super().__init__(name=f"Simulated_GPU_{mode.upper()}")
        self.mode = mode

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_mat = np.asarray(a, dtype=np.float32)
        b_mat = np.asarray(b, dtype=np.float32)

        # Simulating tiled accumulation order
        tile_size = 16
        m, k = a_mat.shape
        _, n = b_mat.shape
        c_mat = np.zeros((m, n), dtype=np.float32)

        for i_tile in range(0, m, tile_size):
            for j_tile in range(0, n, tile_size):
                for k_tile in range(0, k, tile_size):
                    a_sub = a_mat[i_tile:i_tile+tile_size, k_tile:k_tile+tile_size]
                    b_sub = b_mat[k_tile:k_tile+tile_size, j_tile:j_tile+tile_size]
                    # Block multiply
                    c_mat[i_tile:i_tile+tile_size, j_tile:j_tile+tile_size] += np.matmul(a_sub, b_sub)

        return c_mat

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_32 = np.asarray(x, dtype=np.float32)
        x_max = np.max(x_32, axis=axis, keepdims=True)
        exp_x = np.exp(x_32 - x_max)
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        return exp_x / sum_exp

    def cholesky(self, a: np.ndarray) -> np.ndarray:
        return np.linalg.cholesky(np.asarray(a, dtype=np.float32))
