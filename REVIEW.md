# Review guide

What matters when reviewing a MARScope change. Every finding must point to a file and line, and say
what goes wrong for which input. No guesses from names alone.

Start the review summary with the number of findings by severity. If there is nothing to fix, say
so plainly.

## What failure looks like here

MARScope fails quietly. A wrong score raises no error, and a wrong sentence misleads a student or a
journalist. Look for these first.

### Blocking

1. **Preprocessing drifts from training.** Anything that changes decoding (PyAV, RGB), frame stride,
   clip length, padding by repeating the last frame, `n_valid`, the square resize without crop
   (bicubic, antialiased, per channel), or the [-1, 1] scaling. Requires the integration tests' output
   in the PR.
2. **Wrong clip or video score.** The clip score is the model's `fake_prob`. Averaging `frame_probs`
   into a clip score is a bug. The video score is the mean of `fake_prob` over clips.
3. **Mismatched ranges.** A precision paired with the other precision's ranges file (fp16 needs
   `usual_ranges.fp16.json`, fp32 needs `usual_ranges.json`), or a device fallback that does not switch
   ranges.
4. **Silent CPU fallback.** A runner that does not check `session.get_providers()` after creating
   the session, or a fallback that is not reported in the result.
5. **Hard-coded model facts.** Clip length, stride, image size or VRAM peaks written as constants
   instead of read from ONNX metadata or `serving.json`.
6. **A verdict or a threshold.** Any output, UI text, report template or log message that says a
   video *is* real or fake, or any new fixed cut-off on a score. The one allowed summary is the
   reading ("Detectors lean fake"), from the fixed rule in `policy.assess`: it
   describes which way the detectors lean against their usual ranges, never what the video is.
7. **A rule that changes a score**, or policy notes that are not deterministic for the same video
   and versions.
8. **Data leaving the machine.** Frames or the video sent anywhere, including to the LLM. Only the
   structured result and the policy notes may go to the LLM, and only with the user's key.
9. **Unattached citations.** The report must keep only references the policy rules attached.
10. **Raw scores compared across models.** Models are compared by their real rank, never by raw
    probability.
11. **Secrets, model files, videos or dataset clips** added to the repository.

### Important

- Evidence worded as "where the forgery is" instead of "what the model relies on".
- Scope or no-verdict notes missing from a new screen, report or export.
- Versions (checkpoint commit, protocol, policy, package) missing from a result or report.
- New behaviour without unit tests, or tests that need a GPU, network or model files outside
  `-m integration_test`.
- Policy rule changes without a test that the rule fires and a test that it stays silent.
- Public functions without types or docstrings.

### Nits (at most three per review)

Naming, wording, small simplifications. Skip anything ruff or mypy already enforces.

## Paths that always need a human reviewer

- `.github/workflows/` (CI and release)
- Preprocessing, the model runners and score aggregation
- The policy rules and their references
- The report prompt and citation filter

## Skip

- `uv.lock` (review `pyproject.toml` instead)
- Generated files and vendored assets (fonts, logos)
