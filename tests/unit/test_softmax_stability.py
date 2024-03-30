"""Unit tests demonstrating overflow vulnerability vs safe mitigation."""
import pytest
import numpy as np
from numqa.kernels.softmax import naive_unstable_softmax, numerically_stable_softmax


@pytest.mark.unit
def test_softmax_numerical_explosion():
    # In FP32, exp(100) overflows to infinity
    adversarial_logits = np.array([1000.0, 1001.0, 999.0], dtype=np.float32)

    with np.errstate(over='ignore', invalid='ignore'):
        unstable_res = naive_unstable_softmax(adversarial_logits)
    assert np.isnan(unstable_res[0]) or np.isinf(unstable_res[0])

    # Numerically stable version passes without NaNs
    stable_res = numerically_stable_softmax(adversarial_logits)
    assert not np.any(np.isnan(stable_res))
    assert np.isclose(np.sum(stable_res), 1.0, atol=1e-5)
