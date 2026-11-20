# Command line

Analyze a video and print the scores in context:

```bash
uv run --extra cuda marscope --video clip.mp4
```

Choose the models, print every value as JSON, or write a report next to the video:

```bash
uv run --extra cuda marscope --video clip.mp4 --models mars,gend      # MARS alone is the default
uv run --extra cuda marscope --video clip.mp4 --models all
uv run --extra cuda marscope --video clip.mp4 --json
uv run --extra cuda marscope --video clip.mp4 --report pdf          # writes clip-report.pdf
uv run --extra cuda marscope --video clip.mp4 --language fr --jurisdiction eu
```

Start the web UI and REST API instead (see [Web UI and serving](serve.md)):

```bash
uv run --extra cuda marscope --serve
```

## Options

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
| `--verbose` | Log each step |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Done |
| `2` | The file is missing or cannot be read as a video, or an option is wrong (for example an unknown model) |

An analysis of a 10-second clip with all three models takes about 2 to 3 minutes on one GPU,
including loading the models; most of it is the assumption tests.
