"""Softmax implementations: Unsafe (naive) vs Safe (subtracted maximum)."""
import numpy as np


def naive_unstable_softmax(x: np.ndarray) -> np.ndarray:
    """Susceptible to overflow if x contains values > ~88 in FP32."""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def numerically_stable_softmax(x: np.ndarray) -> np.ndarray:
    """Safe implementation: subtracts max value before exponential."""
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)
