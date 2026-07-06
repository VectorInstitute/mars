# deepfake-moe

A clean, minimal foundation for **audio-video deepfake detection with
Qwen3-Omni**. The goal of this repo is *traceability*: load the model once and
run a raw forward pass you can step through with a debugger, with direct access
to intermediate tensors (audio tokens, visual tokens, hidden states).

The MoE / inspection logic is intentionally **not** included yet - this is the
model-access + utilities scaffold to build on.

## Layout

- `deepfake_moe.models.config` - a single dataclass of run settings.
- `deepfake_moe.utils.logging_utils` - one-call logger setup.
- `deepfake_moe.models.hf.qwen_omni` - load the model + processor, build inputs
  *without* the chat template, and run a raw forward pass; plus a separate
  chat-template convenience function.
- `deepfake_moe.utils` - small, single-purpose media helpers (video/audio
  segmentation, media I/O, a transcription stub).
- `deepfake_moe.run_forward` - entry point that runs one raw forward pass on a
  single sample and prints tensor shapes.

## Run one raw forward pass

```bash
python -m deepfake_moe.run_forward \
    --video path/to/clip.mp4 \
    --audio path/to/clip.wav \
    --text "Describe the speaker."
```

See the [user guide](user_guide.md) for development setup.
