"""Property-based testing using Hypothesis to fuzz matrix dimensions and random scales."""
import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from numqa.backends.reference import ReferenceGoldenBackend
from numqa.backends.gpu_simulated import GPUSimulatedBackend
from numqa.core.fp_metrics import FloatingPointMetrics

golden = ReferenceGoldenBackend()
gpu_backend = GPUSimulatedBackend()


@pytest.mark.property
@settings(max_examples=25, deadline=None)
@given(
    dim=st.integers(min_value=4, max_value=32),
    scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
)
def test_property_matrix_multiplication_associativity(dim: int, scale: float):
    rng = np.random.default_rng(int(scale * 100))
    a = (rng.standard_normal((dim, dim)) * scale).astype(np.float32)
    b = (rng.standard_normal((dim, dim)) * scale).astype(np.float32)

    out_gpu = gpu_backend.gemm(a, b)
    out_gold = golden.gemm(a, b)

    drift = FloatingPointMetrics.evaluate_drift(out_gpu, out_gold)
    assert drift["max_rel_error"] < 5e-4
