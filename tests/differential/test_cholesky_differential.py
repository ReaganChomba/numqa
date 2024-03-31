"""Differential test evaluating Cholesky decomposition on well-conditioned and Hilbert matrices."""
import pytest
import numpy as np
from numqa.kernels.cholesky import generate_spd_matrix
from numqa.core.edge_cases import NumericalAdversary
from numqa.core.fp_metrics import FloatingPointMetrics


@pytest.mark.differential
def test_cholesky_well_conditioned(golden_backend):
    spd = generate_spd_matrix(16, condition_number=5.0)
    l_gold = golden_backend.cholesky(spd)

    # Reconstruct matrix: L @ L.T
    recon = l_gold @ l_gold.T
    drift = FloatingPointMetrics.evaluate_drift(recon, spd)
    assert drift["max_rel_error"] < 1e-12


@pytest.mark.differential
def test_cholesky_ill_conditioned_hilbert():
    # Hilbert matrices have astronomical condition numbers
    h = NumericalAdversary.generate_hilbert_matrix(8)
    kappa = FloatingPointMetrics.matrix_condition_number(h)
    assert kappa > 1e8  # Verifies ill-conditioned property

    # Attempting Cholesky in single precision must raise LinAlgError
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(h.astype(np.float32))
