"""Debug summaries for raw thinker forward passes."""

from __future__ import annotations

from typing import Any

import torch


def _shape(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    if isinstance(value, (tuple, list)):
        if value and isinstance(value[0], torch.Tensor):
            return f"tuple[{len(value)}] first={tuple(value[0].shape)}"
        return f"{type(value).__name__}[{len(value)}]"
    return type(value).__name__


def _token_id(processor: Any, token: str) -> int:
    return processor.tokenizer.convert_tokens_to_ids(token)


def count_mm_tokens(input_ids: torch.Tensor, processor: Any) -> dict[str, int]:
    """Count multimodal placeholder tokens in ``input_ids`` (batch 0)."""
    ids = input_ids[0]
    proc = processor
    names = {
        "video": proc.video_token,
        "audio": proc.audio_token,
        "image": proc.image_token,
    }
    counts: dict[str, int] = {}
    for name, token in names.items():
        token_id = _token_id(proc, token)
        counts[name] = int((ids == token_id).sum().item())
    counts["total"] = int(ids.numel())
    counts["other"] = counts["total"] - sum(counts[k] for k in names)
    return counts


def moe_layer_indices(thinker: Any) -> list[int]:
    """Layer indices that use sparse MoE MLP blocks."""
    cfg = thinker.config.text_config
    indices: list[int] = []
    for layer_idx in range(cfg.num_hidden_layers):
        if (layer_idx not in cfg.mlp_only_layers) and (
            cfg.num_experts > 0 and (layer_idx + 1) % cfg.decoder_sparse_step == 0
        ):
            indices.append(layer_idx)
    return indices


def summarize_expert_usage(
    router_indices: torch.Tensor,
    num_experts: int,
) -> dict[str, Any]:
    """Histogram of selected experts from gate ``router_indices`` (seq, top_k)."""
    flat = router_indices.reshape(-1).to(torch.long)
    counts = torch.bincount(flat, minlength=num_experts)
    used = int((counts > 0).sum().item())
    total = int(counts.sum().item())
    top = torch.topk(counts, min(8, num_experts))
    top_pairs = [(int(i), int(c)) for c, i in zip(top.values.tolist(), top.indices.tolist())]
    load = counts.float() / max(total, 1)
    balance = float(load.max() / load[load > 0].min()) if used else 0.0
    return {
        "selections": total,
        "experts_used": used,
        "top_experts": top_pairs,
        "max_min_load_ratio": balance,
    }


def describe_encoder_output(name: str, output: Any) -> list[str]:
    """Human-readable encoder output shapes."""
    lines = [f"  {name}:"]
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        lines.append(f"    last_hidden_state : {tuple(output.last_hidden_state.shape)}")
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        lines.append(f"    pooler_output     : {tuple(output.pooler_output.shape)}")
    if hasattr(output, "deepstack_features") and output.deepstack_features is not None:
        feats = output.deepstack_features
        if isinstance(feats, (tuple, list)):
            lines.append(f"    deepstack_features: {len(feats)} tensors")
            for i, feat in enumerate(feats):
                lines.append(f"      level {i} : {tuple(feat.shape)}")
        else:
            lines.append(f"    deepstack_features: {_shape(feats)}")
    if len(lines) == 1:
        lines.append(f"    (raw) {_shape(output)}")
    return lines


def describe_inputs(inputs: Any, processor: Any) -> list[str]:
    """Processor batch tensor shapes and token breakdown."""
    lines = ["2) Processor batch"]
    skip = {"input_ids", "attention_mask"}
    for key in sorted(inputs.keys()):
        if key in skip:
            continue
        value = inputs[key]
        if isinstance(value, torch.Tensor):
            lines.append(f"  {key:<24} {tuple(value.shape)}")
    if hasattr(inputs, "input_ids"):
        counts = count_mm_tokens(inputs.input_ids, processor)
        lines.append(
            f"  input_ids              {tuple(inputs.input_ids.shape)}  "
            f"(video={counts['video']}, audio={counts['audio']}, "
            f"image={counts['image']}, other={counts['other']})"
        )
    if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
        lines.append(f"  attention_mask         {tuple(inputs.attention_mask.shape)}")
    return lines


def describe_router_capture(
    router_capture: dict[int, Any],
    thinker: Any,
    max_layers: int = 4,
) -> list[str]:
    """Summarize MoE gate outputs keyed by decoder layer index."""
    cfg = thinker.config.text_config
    lines = [
        "6) MoE router (per sparse layer)",
        f"  experts={cfg.num_experts}, top_k={cfg.num_experts_per_tok}, "
        f"sparse_layers={len(moe_layer_indices(thinker))}",
    ]
    if not router_capture:
        lines.append("  (no router outputs captured — gate hooks missing or no MoE layers ran)")
        return lines

    layer_indices = sorted(router_capture)
    for i, layer_idx in enumerate(layer_indices):
        if i >= max_layers:
            lines.append(f"  ... ({len(layer_indices) - i} more sparse layers)")
            break
        gate_out = router_capture[layer_idx]
        if isinstance(gate_out, (tuple, list)):
            logits, scores, indices = gate_out[0], gate_out[1], gate_out[2]
            usage = summarize_expert_usage(indices, cfg.num_experts)
            lines.append(f"  layer {layer_idx:>2}:")
            lines.append(f"    router_logits  {tuple(logits.shape)}  (seq·batch, num_experts)")
            lines.append(f"    router_scores  {tuple(scores.shape)}  (seq·batch, top_k)")
            lines.append(f"    router_indices {tuple(indices.shape)}  (seq·batch, top_k)")
            top = ", ".join(f"e{idx}:{cnt}" for idx, cnt in usage["top_experts"][:5])
            lines.append(
                f"    expert usage   {usage['experts_used']}/{cfg.num_experts} experts, "
                f"top: {top}, load_ratio={usage['max_min_load_ratio']:.2f}"
            )
        else:
            lines.append(f"  layer {layer_idx:>2}: {_shape(gate_out)}")
    return lines


def describe_outputs(
    outputs: Any,
    processor: Any,
    input_ids: torch.Tensor,
) -> list[str]:
    """Thinker output tensors and last-position top tokens."""
    lines = ["7) Thinker output"]
    if outputs.logits is not None:
        lines.append(f"  logits          {tuple(outputs.logits.shape)}  (batch, seq, vocab) — score per token position")
        last_logits = outputs.logits[0, -1].float()
        top = torch.topk(last_logits, 5)
        tokens = processor.tokenizer.convert_ids_to_tokens(top.indices.tolist())
        lines.append("  last position top-5 tokens:")
        for score, token in zip(top.values.tolist(), tokens):
            lines.append(f"    {score:8.3f}  {token!r}")

    if getattr(outputs, "hidden_states", None):
        hs = outputs.hidden_states
        lines.append(f"  hidden_states   {len(hs)} x {tuple(hs[-1].shape)}  (embedding + each decoder layer)")

    if getattr(outputs, "aux_loss", None) is not None:
        lines.append(f"  aux_loss        {float(outputs.aux_loss):.6f}  (load-balancing)")

    return lines


def print_forward_report(  # noqa: PLR0912
    model: Any,
    inputs: Any,
    outputs: Any,
    *,
    raw_media: dict[str, Any] | None = None,
    encoder_capture: dict[str, Any] | None = None,
    router_capture: dict[int, Any] | None = None,
    router_layers_to_show: int = 4,
) -> None:
    """Print end-to-end forward flow from media through encoders, MoE, and logits."""
    thinker = model.thinker
    proc = model.processor
    text_cfg = thinker.config.text_config

    print("\n=== forward pass flow ===")

    print("1) Raw media")
    if raw_media:
        for key, value in raw_media.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key:<16} {tuple(value.shape)}")
            elif isinstance(value, (tuple, list)) and value and hasattr(value[0], "shape"):
                print(f"  {key:<16} {tuple(value[0].shape)}  (1 clip)")
            else:
                print(f"  {key:<16} {_shape(value)}")
    else:
        print("  (not provided)")

    for line in describe_inputs(inputs, proc):
        print(line)

    print("3) Encoders (tower outputs)")
    if encoder_capture:
        for name, out in encoder_capture.items():
            for line in describe_encoder_output(name, out):
                print(line)
    else:
        print("  (no encoder hooks)")

    counts = count_mm_tokens(inputs.input_ids, proc)
    print(
        "4) Sequence merge (thinker inputs_embeds)\n"
        f"  {counts['total']} positions: scatter video ({counts['video']}), "
        f"audio ({counts['audio']}), image ({counts['image']}) features into placeholder tokens"
    )

    moe_layers = moe_layer_indices(thinker)
    print(
        "5) Thinker transformer stack\n"
        f"  layers={text_cfg.num_hidden_layers}, hidden={text_cfg.hidden_size}, "
        f"heads={text_cfg.num_attention_heads}\n"
        f"  MoE sparse layers ({len(moe_layers)}): "
        f"{moe_layers[:8]}{'...' if len(moe_layers) > 8 else ''}\n"
        f"  dense MLP layers: {text_cfg.num_hidden_layers - len(moe_layers)}"
    )

    for line in describe_router_capture(router_capture or {}, thinker, max_layers=router_layers_to_show):
        print(line)

    for line in describe_outputs(outputs, proc, inputs.input_ids):
        print(line)
