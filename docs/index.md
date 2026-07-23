# deepfake-moe

Audio-video deepfake detection with **Qwen3-Omni**. The goal of this repo is
*traceability*: load the model once, run a raw forward pass you can step
through with a debugger (direct access to intermediate tensors — audio
tokens, visual tokens, hidden states), evaluate it zero-shot on public
benchmarks, and fine-tune it via two-stage QLoRA SFT — all from the same
model registry and config-driven entry points.

## Layout

- `deepfake_moe.models` - registry + factory that build a model from
  `configs/models.yaml`; `hf/` (raw forward pass, encoder/router hooks) and
  `vllm/` (tensor-parallel inference) are the two backends.
- `deepfake_moe.eval` - dataset loading, the eval runner, locked-schema JSON
  parsing, and metrics (detection AUC/F1, temporal IoU).
- `deepfake_moe.train` - shared SFT data/collator/targets (`common/`) and the
  two-stage QLoRA SFT pipeline (`qlora_sft/`: Stage 1 thinker LoRA, Stage 2
  frozen-LoRA + modality-projector bridges).
- `deepfake_moe.utils` - env/`.env` loading, logging, and media helpers
  (video/audio segmentation, media I/O).
- `deepfake_moe.run_forward` / `run_eval` / `run_train` - the three CLI entry
  points; each wires config + data + model together and drives one run.

## Run one raw forward pass

```bash
python -m deepfake_moe.run_forward \
    --video path/to/clip.mp4 \
    --audio path/to/clip.wav \
    --text "Describe the speaker."
```

See the [README](https://github.com/VectorInstitute/deepfake-moe#readme) for
eval and training commands, and the [user guide](user_guide.md) for
development setup.
