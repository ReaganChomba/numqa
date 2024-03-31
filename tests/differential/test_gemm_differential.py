"""Differential tests evaluating GEMM across C-Native, Simulated GPU, and Reference backends."""
import pytest
import numpy as np


@pytest.mark.differential
def test_differential_gemm_execution(differential_engine):
    rng = np.random.default_rng(2026)
    m, k, n = 32, 64, 32
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)

    results = differential_engine.run_differential_gemm(
        a, b, max_ulp_tolerance=32, max_rel_error=1e-4
    )

    report = differential_engine.format_report(results)
    print("\n" + report)

    for backend_name, eval_data in results["evaluations"].items():
        assert eval_data["passed"] is True, f"Backend {backend_name} violated numerical tolerance!"
