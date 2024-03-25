"""Pytest fixtures for backends and differential engines."""
import pytest
from numqa.backends.reference import ReferenceGoldenBackend
from numqa.backends.c_native import CNativeBackend
from numqa.backends.gpu_simulated import GPUSimulatedBackend
from numqa.core.differential_engine import DifferentialTestEngine


@pytest.fixture(scope="session")
def golden_backend():
    return ReferenceGoldenBackend()


@pytest.fixture(scope="session")
def differential_engine(golden_backend):
    engine = DifferentialTestEngine(golden_backend)
    engine.register_backend("Native_C_FP32", CNativeBackend())
    engine.register_backend("Simulated_GPU_FP32", GPUSimulatedBackend())
    return engine
