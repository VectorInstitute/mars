# Python API

```python
from marscope import MARScope
from marscope.analysis.ranges import rarity

scope = MARScope(models="mars,gend")      # device="auto", precision="auto", sequential=False, faces=True
result = scope.analyze("clip.mp4", jurisdiction="eu", language="en")

for model in result.models.values():
    print(model.name, f"probability of fake {model.fake:.0%}", rarity(model.context.real_rank, model.name))
    print("  ran on", model.device, model.precision)
    print("  clips:", [round(c.fake, 3) for c in model.clips])

print(result.agreement.strength)          # "consistent", "mixed" or "weak"
for note in result.notes:
    print(note.kind, note.state, note.title)
```

`result` is a Pydantic `VideoResult`; `result.model_dump_json()` gives the same JSON as `--json`.
The models load once per `MARScope` and stay loaded, so reuse the object for several videos.

## Options

| Argument | Values |
|---|---|
| `models` | `"mars,gend"`, a list, `"all"`, or None for MARS only (the default) |
| `device` | `"auto"`, `"cuda"` or `"cpu"` |
| `precision` | `"auto"` (fp16 on the GPU, fp32 on the CPU), `"fp16"` or `"fp32"` |
| `sequential` | Run one model at a time, whatever the free GPU memory |
| `faces` | Detect faces (needed by A1 and the face checks of A3) |

## Tests and reports

```python
from marscope.explain.report import build_report, to_markdown, to_pdf
from marscope.explain.ask import chip_answer

scores = scope.rescore("clip.mp4", result, clip_index=0, drop=[2, 5])   # every model, two runs removed
curve = scope.sweep("clip.mp4", result, "mars", clip_index=0)           # top frames against random

report = build_report(result, use_llm=False, language="en")
print(to_markdown(report))
open("report.pdf", "wb").write(to_pdf(report))

print(chip_answer(result, "tests").text)                              # a fixed answer, no LLM
```

See the [API Reference](../api.md) for every class and function.
