"""Canonical Matrix Multiply definitions and block verification helpers."""
import numpy as np


def naive_gemm(a: np.ndarray, b: np.ndarray, dtype=np.float32) -> np.ndarray:
    a_cast = np.asarray(a, dtype=dtype)
    b_cast = np.asarray(b, dtype=dtype)
    m, k = a_cast.shape
    _, n = b_cast.shape
    c = np.zeros((m, n), dtype=dtype)
    for i in range(m):
        for j in range(n):
            acc = dtype(0.0)
            for p in range(k):
                acc += a_cast[i, p] * b_cast[p, j]
            c[i, j] = acc
    return c
