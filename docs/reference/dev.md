# Development

Set up the environment with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --extra cuda       # or plain `uv sync` without an NVIDIA GPU
source .venv/bin/activate
pre-commit install
```

Checks run before every PR (see [CONTRIBUTING.md](https://github.com/VectorInstitute/marscope/blob/main/CONTRIBUTING.md)):

```bash
uv run pre-commit run --all-files               # ruff, ruff-format, mypy, typos, doctest, unit tests
uv run pytest -m "not integration_test" tests   # unit tests: no models, GPU or network
uv run mypy src
uv run mkdocs build --strict                    # when docs change
```

**Integration tests** (`-m integration_test`) load the real models and check that the pipeline
reproduces the research outputs on fixed clips. They need a GPU and local copies of those clips
(`MARSCOPE_DATA_ROOT`), so they run on a GPU machine, not in CI. Run them before merging any change to
preprocessing, the model runners, score aggregation or the usual ranges. On a Slurm cluster, run
them through the scheduler:

```bash
srun --partition=<gpu-partition> --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
    uv run --extra cuda pytest -m integration_test tests
```

What reviewers check, and what blocks a merge, is in
[REVIEW.md](https://github.com/VectorInstitute/marscope/blob/main/REVIEW.md).

## Build the docs locally

```bash
uv sync --group docs
uv run mkdocs serve
```

Then open `http://127.0.0.1:8000`. If that port is in use (for example by `marscope --serve`), run
`uv run mkdocs serve -a 127.0.0.1:8001`.
