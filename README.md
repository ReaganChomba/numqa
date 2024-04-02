# numqa

**A differential testing framework for numerical correctness across CPU, native C, and GPU-style backends.**

Detects precision drift, IEEE 754 deviations, and mathematical regressions by comparing high-performance kernels against higher-precision reference implementations.

[![CI](https://github.com/ReaganChomba/numqa/actions/workflows/ci.yml/badge.svg)](https://github.com/ReaganChomba/numqa/actions/workflows/ci.yml)

## Table of Contents

- [Features](#features)
- [Backends Under Test](#backends-under-test)
- [Quickstart](#quickstart)
- [Running Tests](#running-tests)
- [Numerical Tolerances and Reproduction](#numerical-tolerances-and-reproduction)
- [Docker](#docker)
- [Continuous Integration](#continuous-integration)
- [Author](#author)

## Features

### Differential oracles

Kernels such as GEMM, Cholesky, and Softmax are checked against FP64 reference implementations. For severely ill-conditioned inputs, where FP64 itself loses significance (condition numbers approaching $1/\varepsilon_{64} \approx 10^{16}$), an arbitrary-precision oracle built on `mpmath` provides ground truth for results and residual norms.

### ULP verification

Floating-point bit patterns are mapped to an ordered integer space so that the integer difference equals the distance in units in the last place (ULP).

| Case | Behavior |
| :-- | :-- |
| Precision | FP32 (`int32` view) and FP64 (`int64` view). When comparing a lower-precision kernel to an FP64 oracle, the oracle is rounded to the kernel's precision first. |
| Signed zeros | Negative values are mapped with `0x80000000 - val` (sign-magnitude), so `+0.0` and `-0.0` are 0 ULP apart. |
| Subnormals | Handled without scaling; steps between `0.0` and the smallest normal are linear. |
| Infinities | Identical infinities are 0 ULP apart. Mismatched infinities fail. |
| NaN | Any comparison involving NaN fails with a sentinel distance. |

### Corner-case fuzzing

- Catastrophic cancellation (loss of significant bits in near-equal subtractions)
- Subnormal underflow
- Ill-conditioned systems ($\kappa(A) > 10^8$)

### Property-based testing

[Hypothesis](https://hypothesis.readthedocs.io/) generates input dimensions and adversarial floating-point scales, then checks invariants such as matrix-product associativity within an error bound (floating-point multiplication is not exactly associative).

## Backends Under Test

| Backend | Library / hardware | Runs in CI | Notes |
| :-- | :-- | :-- | :-- |
| CPU SIMD | NumPy / SciPy (OpenBLAS) | Yes | Kernel selection depends on the host CPU |
| Native C extension | GCC-compiled shared library via `ctypes` | Yes | Built with `-O3 -fPIC -shared` |
| GPU-style kernels | Tiled block emulation / WebGPU (`wgpu-py`) | Yes (emulated) | Tests accumulation order, not real GPU hardware; hardware tests need a self-hosted runner |

## Quickstart

**Prerequisites:** Python 3.10+ and a C compiler (`gcc` or `clang`).

```bash
git clone https://github.com/ReaganChomba/numqa.git
cd numqa
python3 -m venv venv
source venv/bin/activate
pip install -e ".[test]"
```

## Running Tests

Run the full suite with coverage of the framework code:

```bash
pytest -v --cov=numqa --cov-report=term-missing
```

Run individual suites:

```bash
pytest tests/unit/ -v           # IEEE 754 bit conversion and metric unit tests
pytest tests/differential/ -v   # backends vs. FP64 reference
pytest tests/property/ -v       # Hypothesis property-based tests
```

## Numerical Tolerances and Reproduction

**ULP ceilings.** Limits follow rounding-error bounds that grow as $O(K\,\varepsilon_{\mathrm{mach}})$ with the reduction dimension $K$. For FP32 GEMM with $K \le 64$, the maximum allowed drift is 32 ULP, which accounts for reordered tiled accumulation. 

**Compiler flags.** Native C kernels are built with `gcc -O3 -fPIC -shared -lm` and without `-ffast-math`, to keep IEEE 754 rounding semantics. 

**Reproducing a Hypothesis failure.** Hypothesis prints a seed on failure. Re-run the exact case with:

```bash
pytest tests/property/ --hypothesis-seed=<SEED_INTEGER>
```

## Docker

Run the full suite in an isolated Linux container:

```bash
docker build -t numerical-qa-framework .
docker run --rm numerical-qa-framework
```

> **Note:** floating-point results can vary with the CPU and the BLAS kernel chosen at runtime, so a container reduces but does not eliminate variation. The image is based on `python:3.11-slim-bullseye` with Debian's OpenBLAS and GCC. 

## Continuous Integration

Tests run on GitHub Actions via [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

- **Triggers:** pushes and pull requests to `main` and `develop`.
- **Matrix:** Python 3.10 and 3.11 on Ubuntu 22.04.
- **Scope:** builds the native C backend, runs the unit, differential, and property suites, and enforces a coverage threshold.

## Author

Maintained by Reagan ([@ReaganChomba](https://github.com/ReaganChomba)) · thefifthemperor51@gmail.com
