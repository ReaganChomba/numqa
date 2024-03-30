"""Unit tests verifying boundary float values and cancellation pairs."""
import pytest
import numpy as np
from numqa.core.edge_cases import NumericalAdversary


@pytest.mark.unit
def test_boundary_scalars_types():
    scalars = NumericalAdversary.get_ieee754_boundary_scalars(dtype=np.float32)
    assert np.isnan(scalars["nan"])
    assert np.isinf(scalars["infinity_pos"])
    assert scalars["smallest_subnormal"] > 0.0


@pytest.mark.unit
def test_catastrophic_cancellation_detection():
    x, y = NumericalAdversary.generate_catastrophic_cancellation_pair(10, cancellation_bits=23)
    # Subtraction in float32 completely wipes out precision
    diff_f32 = x - y
    diff_exact = np.float64(x[0]) - np.float64(y[0])
    # Relative difference should show catastrophic error
    assert not np.isclose(diff_f32[0], diff_exact, rtol=1e-3)
