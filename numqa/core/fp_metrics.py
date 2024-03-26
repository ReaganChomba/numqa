"""
IEEE 754 Floating-Point Precision and Error Analysis Engine.
Calculates bitwise ULP distances, relative drift, and conditioning.
"""
from typing import Tuple, Dict, Any
import numpy as np


class FloatingPointMetrics:
    @staticmethod
    def calculate_ulp_distance(val_actual: np.ndarray, val_golden: np.ndarray, dtype=np.float32) -> np.ndarray:
        """
        Calculates bitwise Units in the Last Place (ULP) distance between two float arrays.
        Directly interprets bit representations as signed integers.
        """
        act = np.asarray(val_actual, dtype=dtype)
        gold = np.asarray(val_golden, dtype=dtype)

        if act.shape != gold.shape:
            raise ValueError(f"Shape mismatch: {act.shape} vs {gold.shape}")

        if dtype == np.float32:
            int_dtype = np.int32
        elif dtype == np.float64:
            int_dtype = np.int64
        else:
            raise NotImplementedError(f"ULP calculation not supported for {dtype}")

        act_int = act.view(int_dtype)
        gold_int = gold.view(int_dtype)

        # Handle negative zero and sign bit flips in IEEE 754 representation
        act_ulp = np.where(act_int < 0, np.int64(0x80000000 if dtype == np.float32 else 0x8000000000000000) - act_int, act_int)
        gold_ulp = np.where(gold_int < 0, np.int64(0x80000000 if dtype == np.float32 else 0x8000000000000000) - gold_int, gold_int)

        ulp_diff = np.abs(act_ulp - gold_ulp)

        # Replace NaN locations with maximum unsigned distance
        nan_mask = np.isnan(act) | np.isnan(gold)
        ulp_diff = np.where(nan_mask, np.iinfo(int_dtype).max, ulp_diff)

        return ulp_diff

    @staticmethod
    def evaluate_drift(actual: np.ndarray, golden: np.ndarray) -> Dict[str, float]:
        """Calculates absolute error, relative error, and Frobenius norm drift."""
        act = np.asarray(actual, dtype=np.float64)
        gold = np.asarray(golden, dtype=np.float64)

        abs_diff = np.abs(act - gold)
        max_abs_err = float(np.max(abs_diff))

        gold_denom = np.where(np.abs(gold) < 1e-15, 1e-15, np.abs(gold))
        rel_diff = abs_diff / gold_denom
        max_rel_err = float(np.max(rel_diff))

        gold_fro = np.linalg.norm(gold)
        if gold_fro < 1e-15:
            frobenius_rel_err = float(np.linalg.norm(abs_diff))
        else:
            frobenius_rel_err = float(np.linalg.norm(abs_diff) / gold_fro)

        return {
            "max_abs_error": max_abs_err,
            "max_rel_error": max_rel_err,
            "frobenius_rel_error": frobenius_rel_err
        }

    @staticmethod
    def matrix_condition_number(matrix: np.ndarray) -> float:
        """Computes 2-norm matrix condition number kappa(A) = ||A|| * ||A^-1||."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Condition number requires a square 2D matrix.")
        try:
            return float(np.linalg.cond(matrix))
        except np.linalg.LinAlgError:
            return float('inf')
