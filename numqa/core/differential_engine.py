"""
Differential Test Harness: Executes kernels across multiple compute backends,
validates results against a golden oracle, and enforces precision budgets.
"""
from typing import Dict, List, Any
import numpy as np
from tabulate import tabulate
from numqa.core.fp_metrics import FloatingPointMetrics


class DifferentialTestEngine:
    def __init__(self, golden_backend):
        self.golden = golden_backend
        self.backends = {}

    def register_backend(self, name: str, backend_instance):
        self.backends[name] = backend_instance

    def run_differential_gemm(self, a: np.ndarray, b: np.ndarray, max_ulp_tolerance: int = 16, max_rel_error: float = 1e-4) -> Dict[str, Any]:
        """Runs GEMM (A * B) across all backends and evaluates divergence."""
        golden_out = self.golden.gemm(a, b)

        results = {
            "golden_backend": self.golden.name,
            "evaluations": {}
        }

        for name, backend in self.backends.items():
            test_out = backend.gemm(a, b)
            metrics = FloatingPointMetrics.evaluate_drift(test_out, golden_out)
            ulp_distances = FloatingPointMetrics.calculate_ulp_distance(test_out, golden_out, dtype=test_out.dtype)
            max_ulp = int(np.max(ulp_distances))

            passed = (max_ulp <= max_ulp_tolerance) and (metrics["max_rel_error"] <= max_rel_error)

            results["evaluations"][name] = {
                "passed": passed,
                "max_ulp": max_ulp,
                "metrics": metrics,
                "output_dtype": str(test_out.dtype)
            }

        return results

    @staticmethod
    def format_report(results: Dict[str, Any]) -> str:
        """Generates an ASCII verification compliance table."""
        table_rows = []
        for backend_name, data in results["evaluations"].items():
            metrics = data["metrics"]
            status = "PASSED" if data["passed"] else "FAILED"
            table_rows.append([
                backend_name,
                data["output_dtype"],
                status,
                data["max_ulp"],
                f"{metrics['max_abs_error']:.3e}",
                f"{metrics['max_rel_error']:.3e}",
                f"{metrics['frobenius_rel_error']:.3e}"
            ])

        headers = ["Backend", "DType", "Status", "Max ULP", "Max Abs Err", "Max Rel Err", "Frob Rel Err"]
        return tabulate(table_rows, headers=headers, tablefmt="grid")
