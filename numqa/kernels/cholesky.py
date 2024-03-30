"""Symmetric Positive-Definite (SPD) Matrix generation & checks."""
import numpy as np


def generate_spd_matrix(n: int, condition_number: float = 10.0, dtype=np.float64) -> np.ndarray:
    """Generates an n x n symmetric positive definite matrix with controlled condition number."""
    rng = np.random.default_rng(42)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigenvalues = np.linspace(1.0, condition_number, n)
    spd = q @ np.diag(eigenvalues) @ q.T
    return spd.astype(dtype)
