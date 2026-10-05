# Installation

The [README](../README.md#installation) gives the install commands and time. This page adds the
CuPy package for your GPU driver, checks, extras and fixes for common problems. To run the tests,
see [Running the tests](../README.md#running-the-tests).

## Choosing a CuPy package

GPU support needs CuPy built for the CUDA version that your NVIDIA driver supports. Find that
version in the header of `nvidia-smi`:

<!-- docs-test: skip (needs an NVIDIA driver) -->
```bash
nvidia-smi
```

```text
| NVIDIA-SMI 565.57.01              Driver Version: 565.57.01      CUDA Version: 12.7     |
```

This is the newest CUDA the driver supports (not an installed CUDA Toolkit); the driver also runs
CuPy built for an older CUDA, so minor versions need not match.

| Driver's CUDA version | Install command | Tested |
|---|---|---|
| 12.x or newer | `pip install ".[gpu]"` (installs `cupy-cuda12x`) | Yes (`cupy-cuda12x` 14.2.0) |
| 12.x or newer, with conda | `conda install -c conda-forge cupy`, then `pip install .` | Yes (CuPy 14.0.1) |
| 13.x, only if you need CUDA 13 | `pip install . cupy-cuda13x` | No. CUDA 13 does not support Pascal or older GPUs, such as the tested TITAN X |
| 11.x or older | Not supported: update the driver | |

- **One CuPy package per environment.** With conda-forge CuPy, install DUET with `pip install .`,
  not `".[gpu]"`, which would add `cupy-cuda12x` next to it. On a driver that reports CUDA 13,
  add `"cuda-version=12"` to the conda command to stay on CUDA 12.
- **CUDA libraries.** The `cupy-cuda12x` wheel loads NVRTC and cuBLAS from a CUDA Toolkit 12 on the
  machine (the tested machine has 12.6). Without one, use `pip install "cupy-cuda12x[ctk]"` (not
  tested) or conda-forge CuPy, which brings its own. Other setups: the
  [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

## Checking the installation

From the root of your clone, in the activated environment:

```bash
python -c "import duet; print(duet.__version__)"
```

```text
0.2.0
```

`0+unknown` means DUET was imported from the source tree without being installed. Next, check
whether DUET sees a GPU:

```bash
python -c "from duet.gpu_utils import has_gpu, get_gpu_count; print(has_gpu(), get_gpu_count())"
```

```text
False 0
```

On a GPU machine this prints `True` and the number of visible GPUs, which `device="auto"` uses.
To test CuPy itself (it prints the CuPy version, then `3`):

<!-- docs-test: skip (needs CuPy and an NVIDIA GPU) -->
```bash
python -c "import cupy; print(cupy.__version__); print(cupy.arange(3).sum())"
```

## Extras and development install

| Extra | Adds | Needed for |
|---|---|---|
| `gpu` | `cupy-cuda12x>=14.0.1` | the PEP and the evaluation on NVIDIA GPUs |
| `benchmark` | pyarrow, pymoo, pyyaml, requests, scikit-learn, seaborn | `scripts/benchmark/` and `experiments/`; not the public API |
| `notebooks` | nbconvert, nbformat, ipykernel | the notebook of `experiments/nisseq_error_analysis/`, run headless |
| `test` | pytest | the test suite |
| `all` | all of the above | |

To install exactly the tested versions, add the constraints file of the CI "oldest" job:
`pip install -c ci/constraints-oldest.txt ".[benchmark,test]"`. An editable install loads DUET from
`src/duet/` in your clone; the scripts in `experiments/` require it. `environment.yml` creates a
CPU-only environment `duet` with that install:

<!-- docs-test: skip (installs packages) -->
```bash
conda env create -f environment.yml   # or: pip install -e ".[benchmark,test]"
conda activate duet
pip install -e ".[gpu]"               # keep -e when you add an extra
```

Reproducing the paper also needs two auxiliary conda environments, `ops` (the Feldman et al. 2019
baseline) and `scanpy_env` (the brain-atlas scripts), from `environments/`; see
[Reproducing the paper](reproducing_the_paper.md#setup).

## Troubleshooting

| Symptom | Fix |
|---|---|
| The GPU check prints `False 0` | Run `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"` to see the CUDA error: DUET treats every CUDA error as "no GPU". |
| `ModuleNotFoundError: No module named 'cupy'` | Install the `gpu` extra from your clone: `pip install ".[gpu]"`. DUET is not on PyPI, so the `pip install "duet-codebook[gpu]"` that some messages print does not work. |
| `cudaErrorInsufficientDriver` | The CuPy package needs a newer driver (for example `cupy-cuda13x` on a CUDA 12 driver). Uninstall it and pick one from the [table](#choosing-a-cupy-package), or update the driver. |
| `cudaErrorNoDevice` | No GPU is visible: `CUDA_VISIBLE_DEVICES` hides them (unset it), or the container or batch job has no GPU access (for example `docker run --gpus all`). |
| CuPy warns about two packages | Keep one: `python -m pip list \| grep -i cupy`, then uninstall the others. |
| The GPU check prints `True`, but the first GPU computation fails to load NVRTC or cuBLAS | Missing CUDA libraries: install a CUDA Toolkit 12, `"cupy-cuda12x[ctk]"` or conda-forge CuPy. |
| pip: `requires a different Python: 3.10.x not in '>=3.11'` | Create the environment with Python 3.11, 3.12 or 3.13. |
| An error about the `license` field while building | Building needs setuptools 77 or later. Upgrade pip (`python -m pip install --upgrade pip`); without internet access, `pip install "setuptools>=77"` and then `pip install --no-build-isolation .`. |
| `ERROR: the active env imports duet from ..., not from .../src/duet` (experiment scripts) | Activate the right environment and use an editable install from this clone, `pip install -e ".[benchmark,test]"`. |
| `ModuleNotFoundError: No module named 'duet'` | The environment is not activated, or `python` is another interpreter: `python -m pip --version` shows which environment pip belongs to. |

For GPU problems during a run, see [Troubleshooting GPUs](gpu_and_scaling.md#troubleshooting-gpus).
