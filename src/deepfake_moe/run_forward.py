"""CLI: raw thinker forward pass with optional text generation."""

from __future__ import annotations

import argparse

from deepfake_moe.models import QwenOmni
from deepfake_moe.models.config import Config
from deepfake_moe.models.hf.inspect_forward import print_forward_report
from deepfake_moe.models.registry import load_config
from deepfake_moe.utils.logging_utils import get_logger
from deepfake_moe.utils.media_io import load_audio, load_video_frames


logger = get_logger("deepfake_moe.run_forward")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one raw Qwen3-Omni forward pass.")
    parser.add_argument(
        "--model",
        default="qwen3-omni-hf",
        help="Model name in configs/models.yaml (must be an hf backend).",
    )
    parser.add_argument("--models-config", default="configs/models.yaml", help="Model registry YAML.")
    parser.add_argument("--video", default=None, help="Path to a video file.")
    parser.add_argument("--audio", default=None, help="Path to an audio file.")
    parser.add_argument("--text", default="Describe what is happening.", help="Text prompt.")
    parser.add_argument("--model-id", default=None, help="Override config.model_id.")
    parser.add_argument(
        "--router-layers",
        type=int,
        default=4,
        help="MoE layers to detail in the router summary (default: 4).",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="After raw forward, run generate() and print the text answer.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Skip raw forward; only run generate().",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override config max_new_tokens for generate.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    if args.generate_only:
        args.generate = True

    config = load_config(args.model, args.models_config)
    if not isinstance(config, Config):
        raise SystemExit(
            f"run_forward requires an hf-backend model; {args.model!r} is {getattr(config, 'backend', '?')}"
        )
    if args.model_id:
        config.model_id = args.model_id

    model = QwenOmni(config).load()

    raw_media: dict[str, object] = {}
    videos = None
    audios = None
    if args.video:
        frames = load_video_frames(args.video, config.video_max_frames)
        videos = [frames]
        raw_media["video_frames"] = frames
    if args.audio:
        waveform = load_audio(args.audio, config.audio_sample_rate)
        audios = [waveform]
        raw_media["audio_waveform"] = waveform

    if not args.generate_only:
        inputs = model.build_raw_inputs(args.text, audio=audios, videos=videos)
        logger.info("Input ids shape: %s", tuple(inputs.input_ids.shape))

        encoder_capture, enc_handles = model.encoder_hooks()
        router_capture, router_handles = model.router_hooks()
        handles = enc_handles + router_handles
        try:
            outputs = model.raw_forward(inputs, output_router_logits=True)
        finally:
            for handle in handles:
                handle.remove()

        print_forward_report(
            model,
            inputs,
            outputs,
            raw_media=raw_media,
            encoder_capture=encoder_capture,
            router_capture=router_capture,
            router_layers_to_show=args.router_layers,
        )

    if args.generate:
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else config.max_new_tokens
        logger.info("Running generate (max_new_tokens=%d)", max_new_tokens)
        answer = model.generate(
            args.text,
            video=args.video,
            audio=args.audio,
            max_new_tokens=max_new_tokens,
        )
        print("\n=== generated text ===")
        print(answer)


if __name__ == "__main__":
    main()
