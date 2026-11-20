# Installation

MARScope is not on PyPI yet. Install it from the repository with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/VectorInstitute/marscope.git
cd marscope
uv sync --extra cuda       # NVIDIA GPU on Linux or Windows
uv sync                    # Apple silicon Mac, or a machine without an NVIDIA GPU
```

Then run commands with `uv run --extra cuda marscope ...` (or `uv run marscope ...` without a GPU).
Keeping `--extra cuda` on `uv run` means a later update of the lock file still installs the CUDA
libraries.

Once MARScope is released: `pip install marscope[cuda]` (GPU) or `pip install marscope`.

## Where it runs

| Machine | Runs on |
|---|---|
| Linux x86_64 or Windows x64 with an NVIDIA GPU (laptop, workstation, cluster node) | GPU, fp16 |
| Apple silicon Mac, ARM Linux, Windows on ARM | CPU, fp32 |
| Any machine without an NVIDIA GPU | CPU, fp32 |

- **Python** 3.12 to 3.14.
- **GPU memory:** 2.3–2.9 GB per model in fp16, so all three fit on an 8 GB card. `--sequential`
  runs one model at a time.
- **CPU:** works everywhere ONNX Runtime does, but much more slowly; the result says when a model
  ran on the CPU.
- **Intel Macs** are not supported: ONNX Runtime no longer publishes wheels for them.
- **Google Colab** is not tested yet.

## The `cuda` extra

ONNX Runtime's GPU build needs the CUDA 12 and cuDNN 9 libraries. The `cuda` extra installs them as
Python wheels, so nothing has to be installed system-wide, and any NVIDIA driver from CUDA 12 on
works (newer drivers too). On a machine that already has CUDA 12 and cuDNN 9 installed, plain
`uv sync` is enough.

If CUDA cannot be loaded, the models fall back to the CPU and you see a warning such as:

```text
Failed to load libcublasLt.so.12: libcublasLt.so.12: cannot open shared object file
WARNING MARS: CUDA could not be loaded (CUDA 12 or cuDNN 9 libraries missing), so the model runs on the CPU with the fp32 file
```

The fix is `uv sync --extra cuda`, then start again with `uv run --extra cuda ...`.

## Check that the GPU is used

- The app's start screen shows "GB of GPU memory free" (or "No GPU found").
- Every score shows where it ran, for example `cuda fp16`.
- `uv run --extra cuda marscope --video clip.mp4 --json` has `"device": "cuda"` for each model.

## Where files go

| What | Where | Change it with |
|---|---|---|
| Models | The Hugging Face cache (`~/.cache/huggingface`) | `HF_HOME` |
| Face detector and other small files | `~/.cache/marscope` | `MARSCOPE_CACHE` |
| A job's video and frames (server) | A temporary folder, deleted with the job | |

## Optional: a language model for the report

See [Reports and Ask](report.md).
