# MARScope

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/marscope/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/marscope/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/marscope/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/marscope/actions/workflows/unit_tests.yml)
[![docs](https://github.com/VectorInstitute/marscope/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/marscope/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/marscope/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/VectorInstitute/marscope)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/marscope)

**See what a video deepfake detector sees, and how far to trust it.**

MARScope runs up to three trained video deepfake detectors ([MARS](https://github.com/VectorInstitute/mars), GenD and DFD-HR) on a video, on
your own machine. It shows the evidence behind each score, lets you test what each model relies on,
and explains the result with fixed, cited notes. It is a study tool: it never says a video is real
or fake.


## Overview

Video → frames (as in training) → each model's score per clip and for the whole video → the score
placed among the scores the model gives real and fake test videos → tests of what the model relies
on → notes (trust, provenance, laws that may apply) → optional report.

- **Scores in context.** "66%" means little alone. Each score is shown next to the range the model
  usually gives real videos ("very unusual for a real video: fewer than 1 in 100 real videos score
  this high"), drawn as a bar, and models are compared by
  that rank, never by raw score.
- **Evidence.** MARS has built-in maps of the frames and areas it relies on. For GenD and DFD-HR,
  an occlusion map shows which areas change the score when hidden.
- **Tests.** Hide a model's top frames or areas and compare with hiding as many at random. Edit
  the clip in *What if* (JPEG quality, resolution, blur, noise) and score it again.
- **Assumption checks (A1 to A5).** Each analysis asks, for each model: does the score break when
  the face outline is hidden (A1), when the video is shared many times (A4), when fine pixel detail
  is removed (A5)? A2 and A3 come from the model's results on new generators and from checks on the
  video.
- **Notes with references.** Deterministic rules write the notes from the measured values, with
  references. Rules never change a score.
- **Report.** Markdown, HTML or PDF, with what you tried in the session. A language model can write
  the summary prose (optional); every number and reference comes from the template.
- **Local.** The video and its frames never leave the machine. Only the structured result goes to
  the language model, and only if you set one up.

## Requirements

- Python 3.12 to 3.14, [uv](https://docs.astral.sh/uv/getting-started/installation/).
- An NVIDIA GPU is recommended. Peak memory per model is 2.3–2.9 GB in fp16 (the default on GPU);
  all three fit on an 8 GB card, and `--sequential` runs one at a time. Without a GPU the models
  run on the CPU (fp32), much more slowly, and the result says so.
- The models (about 0.8 GB each in fp16) download from Hugging Face on first use and are cached.

## Install

```bash
git clone https://github.com/VectorInstitute/marscope.git
cd marscope
uv sync --extra cuda       # NVIDIA GPU on Linux or Windows (then: uv run --extra cuda marscope ...)
uv sync                    # Apple silicon Mac, a machine without an NVIDIA GPU, or with CUDA 12 installed
```

The `cuda` extra installs the CUDA 12 and cuDNN 9 libraries ONNX Runtime needs as Python wheels, so
nothing has to be installed system-wide. Any NVIDIA driver from CUDA 12 on works. If CUDA cannot be loaded, the models run on the CPU
and the result says so (`device` in the JSON, a note on the start screen and on each score).

| Where | Runs on |
|---|---|
| Linux x86_64 or Windows x64 with an NVIDIA GPU | GPU, fp16 |
| Apple silicon Mac, ARM Linux, Windows on ARM, any machine without an NVIDIA GPU | CPU, fp32 (slow) |

Python 3.12 to 3.14. Intel Macs are not supported (ONNX Runtime no longer publishes wheels for them).
With pip instead of uv: `pip install marscope[cuda]` (GPU) or `pip install marscope`.

## Quick start

```bash
uv run --extra cuda marscope --serve         # open http://localhost:8000/app (drop --extra cuda without a GPU)
uv run marscope --video clip.mp4             # scores in context, in the terminal
```

## Command line

```bash
uv run marscope --video clip.mp4                           # MARS (the default)
uv run marscope --video clip.mp4 --models all              # all three models
uv run marscope --video clip.mp4 --models mars,gend        # a subset
uv run marscope --video clip.mp4 --json                    # every value, as JSON
uv run marscope --video clip.mp4 --report pdf              # writes clip-report.pdf next to the video
uv run marscope --video clip.mp4 --language fr --jurisdiction eu
```

| Option | What it does |
|---|---|
| `--video PATH` / `--serve` | Analyze one file, or start the web UI and REST API |
| `--models` | `mars` (default), `gend`, `dfd-hr`, comma-separated, or `all` |
| `--sequential` | One model at a time, for a small GPU |
| `--device` | `auto` (default), `cuda` or `cpu` |
| `--precision` | Model file on the GPU: `auto` (fp16, default) or `fp32`. The CPU always runs fp32 |
| `--no-faces` | Skip face detection (A1 and the face checks of A3 are then skipped) |
| `--json` | Print the full result as JSON |
| `--report` | `markdown`, `html` or `pdf`, written next to the video |
| `--no-llm` | Use the template summary even if a language model is set up |
| `--language` | `en` (default) or `fr`, for notes and reports |
| `--jurisdiction` | `eu`, `ca`, `us` or `all`: which laws the context notes mention |
| `--host`, `--port` | Where `--serve` listens (default `127.0.0.1:8000`) |

## Web UI

```bash
uv run marscope --serve
```

Open **http://localhost:8000/app** (the landing page is at `/`). Choose a video (up to 2 minutes and
1080p), pick the models, and press **Analyze**. The result has four tabs: **Check** (evidence and
tests), **Notes**, **What if** and **Learn**, plus the **Report** sheet. By default you are asked to
guess real or fake before any score is shown.

By default the server listens only on `127.0.0.1`, so nothing outside your machine can reach it.

### On a remote GPU machine or a cluster

Run the server on the GPU machine and let it listen on its network interface:

```bash
uv run --extra cuda marscope --serve --host 0.0.0.0
```

Then, on your laptop, forward a local port to it, through the login node if there is one:

```bash
ssh -N -L 8010:<gpu-node>:8000 <user>@<login-host>
```

and open **http://localhost:8010/app**. After you log in, the command stays silent: the tunnel is
open until you stop it.

- `<gpu-node>` is the machine the server runs on (on Slurm, the node name `squeue` shows).
- **"Address already in use"** means something on your laptop already holds that local port, often
  an older tunnel still running in another terminal, or your editor's port forwarding. Pick another
  local port (`-L 8020:...`), or find it with `lsof -nP -iTCP -sTCP:LISTEN` and close it.
- VS Code and Cursor can forward the port for you from the **Ports** tab instead.
- On a shared machine, `--host 0.0.0.0` lets other users of the network reach the UI; jobs and
  their videos are deleted an hour after last use, or when you leave the result.

## Python API

```python
from marscope import MARScope
from marscope.analysis.ranges import rarity

scope = MARScope(models="mars,gend")      # models=None runs MARS only; "all" runs all three              # device="auto", sequential=False
result = scope.analyze("clip.mp4", language="en")

for model in result.models.values():
    print(model.name, f"probability of fake {model.fake:.0%}", rarity(model.context.real_rank, model.name))
for note in result.notes:
    print(note.kind, note.title, "-", note.text)
```

`result` is a Pydantic `VideoResult` (`result.model_dump_json()` gives the same JSON as `--json`).
Reports:

```python
from marscope.explain.report import build_report, to_markdown

print(to_markdown(build_report(result, use_llm=False)))
```

## REST API

`--serve` also exposes the API the UI uses; the interactive docs are at
**http://localhost:8000/api/docs**. The main calls:

| Call | What it does |
|---|---|
| `GET /health` | Status, GPU and free memory |
| `POST /jobs` | Upload a video (form fields `video`, `models`, `jurisdiction`, `language`); returns a job id |
| `GET /jobs/{id}/events` | Progress as server-sent events |
| `GET /jobs/{id}` | The result |
| `POST /jobs/{id}/report` | The report as JSON, Markdown, HTML or PDF |
| `DELETE /jobs/{id}` | Delete the video and the result now |

## Report summary with a language model (optional)

Without a language model, reports use a fixed template. To have one write the summary prose, add an
OpenAI-compatible endpoint to a `.env` file in the folder you run MARScope from (`cp .env.example .env`):

```bash
MARSCOPE_LLM_BASE_URL=https://your-endpoint/v1
MARSCOPE_LLM_API_KEY=...
MARSCOPE_LLM_MODEL=model-name
```

Only the structured result, the notes and a plain description of your tests are sent, never the
video or frames. Sentences with numbers or a conclusion about the video, and references the rules did not attach, are
removed; if the call fails, the template is used.

**Ask** (in the Learn tab) answers general questions ("What is a deepfake?", "What is C2PA?") from a
small set of reviewed knowledge cards written from the cited research papers, with no language model needed.
Questions about a result use the language model, with the same rules as the report.

## Scope and limits

- **For learning.** Scores are model outputs and can be wrong. Legal, forensic, journalistic and
  employment decisions, and claims about a person, need qualified experts and other evidence.
- The models were trained on face videos (FF++) and rendered game footage (GTA-V). Scores on videos
  unlike that (no faces, new generators, heavy compression, very short clips) are less reliable,
  and the notes say so.
- Video only: no audio, no images.

## Developing

```bash
uv sync                    # runtime, dev and docs groups
source .venv/bin/activate
pre-commit install
```

Checks run before every PR (see [CONTRIBUTING.md](CONTRIBUTING.md)):

```bash
uv run pre-commit run --all-files            # ruff, mypy, typos, doctest, unit tests
uv run pytest -m "not integration_test" tests
```

Docs: `uv run mkdocs serve` (sources in `docs/`).

## Citation

MARScope runs MARS ([code](https://github.com/VectorInstitute/mars), [project page](https://vectorinstitute.github.io/mars/),
[checkpoint](https://huggingface.co/vector-institute/mars)). If you use MARS, please cite:

```bibtex
@misc{radwan2026marsmultiscaleappearanceresidualstreams,
  title         = {MARS: Multiscale Appearance-Residual Streams for Video Forgery Detection},
  author        = {Ahmed Y. Radwan and Shaina Raza},
  year          = {2026},
  eprint        = {2611.12508},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2611.12508}
}
```

## Team

Developed by the **AI Engineering** team at the [Vector Institute](https://vectorinstitute.ai). MARScope is
licensed under the Apache License 2.0. See [LICENSE.md](./LICENSE.md) in the repository. The example frame on
the landing page is from [GenVideo](https://modelscope.cn/datasets/cccnju/Gen-Video) ([DeMamba](https://github.com/chenhaoxing/DeMamba)),
used under CC BY-NC 4.0, with highlights added by MARScope.

For research collaborations, partnerships, or technical inquiries, please contact **Shaina Raza, PhD** at
shaina.raza@vectorinstitute.ai.

## Acknowledgement

Resources used in preparing this research are provided, in part, by the Province of Ontario, the Government of
Canada through CIFAR, and companies sponsoring the Vector Institute.

This research is also supported by the European Union's Horizon Europe research and innovation programme under
the AIXPERT project (Grant Agreement No. 101214389).

## Support

- Open an issue on GitHub: [https://github.com/VectorInstitute/marscope/issues](https://github.com/VectorInstitute/marscope/issues)
- For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md)
