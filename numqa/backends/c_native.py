"""
Native C Compute Backend: Compiles C-level floating point loops via GCC/Clang
on-the-fly and loads them via ctypes to simulate native execution engines.
"""
import ctypes
import os
import subprocess
import tempfile
import numpy as np
from numqa.backends.base import ComputeBackend


C_KERNEL_SOURCE = r"""
#include <math.h>

void c_gemm_fp32(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void c_softmax_fp32(const float* X, float* Out, int N) {
    float max_val = -1e30f;
    for (int i = 0; i < N; ++i) {
        if (X[i] > max_val) max_val = X[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < N; ++i) {
        Out[i] = expf(X[i] - max_val);
        sum_exp += Out[i];
    }
    for (int i = 0; i < N; ++i) {
        Out[i] /= sum_exp;
    }
}
"""


class CNativeBackend(ComputeBackend):
    def __init__(self):
        super().__init__(name="Native_C_FP32")
        self._lib = self._compile_and_load()

    def _compile_and_load(self):
        temp_dir = tempfile.mkdtemp()
        src_file = os.path.join(temp_dir, "kernel.c")
        lib_file = os.path.join(temp_dir, "kernel.so")

        with open(src_file, "w") as f:
            f.write(C_KERNEL_SOURCE)

        cmd = ["gcc", "-O3", "-fPIC", "-shared", "-o", lib_file, src_file, "-lm"]
        subprocess.check_call(cmd)

        lib = ctypes.CDLL(lib_file)

        # Setup signatures
        lib.c_gemm_fp32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.c_softmax_fp32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        return lib

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_f32 = np.ascontiguousarray(a, dtype=np.float32)
        b_f32 = np.ascontiguousarray(b, dtype=np.float32)
        m, k = a_f32.shape
        k_b, n = b_f32.shape
        assert k == k_b, "Inner dimensions must match"

        c_f32 = np.zeros((m, n), dtype=np.float32)

        ptr_a = a_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ptr_b = b_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ptr_c = c_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._lib.c_gemm_fp32(ptr_a, ptr_b, ptr_c, m, k, n)
        return c_f32

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_f32 = np.ascontiguousarray(x, dtype=np.float32).flatten()
        n = x_f32.shape[0]
        out_f32 = np.zeros(n, dtype=np.float32)

        ptr_x = x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ptr_out = out_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._lib.c_softmax_fp32(ptr_x, ptr_out, n)
        return out_f32.reshape(x.shape)

    def cholesky(self, a: np.ndarray) -> np.ndarray:
        # Fallback to NumPy float32 for linear solve
        return np.linalg.cholesky(a.astype(np.float32))
