"""
IEEE 754 Corner Case & Adversarial Matrix Generator.
Supplies subnormals, denormals, signed zeroes, infinities, and ill-conditioned systems.
"""
from typing import Tuple
import numpy as np


class NumericalAdversary:
    @staticmethod
    def get_ieee754_boundary_scalars(dtype=np.float32) -> Dict[str, float]:
        """Provides IEEE 754 edge-case scalar values for the given precision."""
        finfo = np.finfo(dtype)
        return {
            "zero_pos": dtype(0.0),
            "zero_neg": dtype(-0.0),
            "smallest_subnormal": finfo.smallest_subnormal,
            "smallest_normal": finfo.tiny,
            "max_representable": finfo.max,
            "infinity_pos": dtype(float('inf')),
            "infinity_neg": dtype(float('-inf')),
            "nan": dtype(float('nan'))
        }

    @staticmethod
    def generate_catastrophic_cancellation_pair(n: int, cancellation_bits: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produces arrays x, y such that x - y suffers severe mantissa cancellation.
        x and y agree up to 'cancellation_bits'.
        """
        scale = 1e6
        x = np.ones(n, dtype=np.float64) * scale
        delta = scale / (2.0 ** cancellation_bits)
        y = x - delta
        return x.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def generate_hilbert_matrix(n: int, dtype=np.float64) -> np.ndarray:
        """
        Generates an n x n Hilbert matrix: H_{i,j} = 1 / (i + j + 1).
        Classic pathological ill-conditioned matrix for testing linear solvers.
        """
        i, j = np.indices((n, n))
        h = 1.0 / (i + j + 1.0)
        return h.astype(dtype)

    @staticmethod
    def generate_denormal_vector(length: int, dtype=np.float32) -> np.ndarray:
        """Generates vectors containing subnormal numbers to verify FTZ/DAZ behavior."""
        finfo = np.finfo(dtype)
        subnormals = np.linspace(finfo.smallest_subnormal, finfo.tiny * 0.9, length, dtype=dtype)
        return subnormals
