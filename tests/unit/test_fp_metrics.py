"""Unit test suite for IEEE 754 bitcast ULP calculations."""
import pytest
import numpy as np
from numqa.core.fp_metrics import FloatingPointMetrics


@pytest.mark.unit
def test_ulp_identical_arrays():
    a = np.array([1.0, 2.0, -3.5], dtype=np.float32)
    b = np.array([1.0, 2.0, -3.5], dtype=np.float32)
    ulp = FloatingPointMetrics.calculate_ulp_distance(a, b, dtype=np.float32)
    np.testing.assert_array_equal(ulp, [0, 0, 0])


@pytest.mark.unit
def test_ulp_single_step_diff():
    a = np.array([1.0], dtype=np.float32)
    # Next representable float after 1.0 in FP32 has bit difference of 1
    next_float = np.nextafter(a, np.array([2.0], dtype=np.float32))
    ulp = FloatingPointMetrics.calculate_ulp_distance(a, next_float, dtype=np.float32)
    assert ulp[0] == 1


@pytest.mark.unit
def test_matrix_condition_number_identity():
    ident = np.eye(4, dtype=np.float64)
    kappa = FloatingPointMetrics.matrix_condition_number(ident)
    assert np.isclose(kappa, 1.0)
