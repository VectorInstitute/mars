# Contributing to MARScope

## Workflow

1. Open or pick an issue. If the change alters direction or scope, discuss it in the issue first,
   not in review of a finished PR.
2. Branch from `main`: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `chore/<topic>`.
3. Keep PRs small: one concern per PR, easy to review and easy to revert.
4. Run the checks below and paste their output, or a summary of it, in the PR's **Testing** section.
   Say what you ran, not just that it passed.
5. Fill in the PR template, including **Risk** and **How to revert**.
6. Resolve every review thread before merging: fix it, or close it with a short reason.
7. You land it, you own it: if your change breaks something, you fix it or revert it.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat: ...`, `fix: ...`, `docs: ...`, `test: ...`, `chore: ...`.

## Checks

```bash
uv sync
uv run pre-commit run --all-files               # ruff, ruff-format, mypy, typos, doctest, unit tests
uv run pytest -m "not integration_test" tests   # unit tests
uv run mypy src                                 # types
uv run mkdocs build --strict                    # docs, when docs change
```

**Unit tests** need no models, no GPU and no network. They run in GitHub CI on every PR.

**Integration tests** (`-m integration_test`) load the real ONNX models and check that the pipeline
reproduces the research code's outputs on fixed clips. They need a GPU and local copies of those
clips (research-only datasets, never committed), so they run on the cluster, not in GitHub CI. Run them
before merging any change to preprocessing, the model runners, score aggregation or the usual ranges,
and paste the output in the PR.

On a Slurm cluster, run tests through the scheduler, never on a login node, for example:

```bash
srun --partition=<gpu-partition> --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
    uv run pytest -m integration_test tests
```

## Conventions

- Python 3.12, `src/` layout, managed with `uv`. Add dependencies with `uv add`; never edit
  `uv.lock` by hand.
- Type hints everywhere; `mypy` runs in strict mode.
- [numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) on every public
  function and class.
- [ruff](https://docs.astral.sh/ruff/) for linting and formatting (line length 120).
- Model facts (clip length, frame stride, image size) come from each model's ONNX metadata and
  `serving.json`, never from constants in the code.
- No model files, videos or dataset clips in the repository. Test fixtures are small and synthetic.
- User-facing text never states a verdict ("this video is fake"). It gives likelihoods, and it
  describes evidence as what the model relies on, not where a forgery is.

## Review

What reviewers check, and what blocks a merge, is in [REVIEW.md](REVIEW.md).
