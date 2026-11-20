## Summary

<!-- What this PR does and why, in two or three sentences. Link the issue: "Closes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation
- [ ] Refactoring (no behaviour change)
- [ ] Tests or CI

## Testing

<!-- What you ran and what it printed (paste the output or a summary). Not just "tests pass". -->

- [ ] `uv run pre-commit run --all-files`
- [ ] `uv run pytest -m "not integration_test" tests`
- [ ] Integration tests on a GPU (required for preprocessing, runners, aggregation or ranges)
- [ ] Manual check (describe below; screenshots for UI changes)

## Risk

<!-- What could go wrong, and what the checks above would not catch. Low / medium / high. -->

## How to revert

<!-- "Revert this PR" is fine if true. Note anything a revert does not undo (released packages, caches). -->

## Checklist

- [ ] Small and focused on one concern
- [ ] No verdict wording; evidence is "what the model relies on" (see REVIEW.md)
- [ ] No model files, videos, dataset clips or secrets added
- [ ] Docs updated if behaviour changed
