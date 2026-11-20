"""Command line: ``marscope --video clip.mp4`` or ``marscope --serve``."""

import argparse
import logging
import math
import os
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from marscope.analysis.ranges import rarity
from marscope.explain.policy import JURISDICTIONS
from marscope.models.registry import parse_models


if TYPE_CHECKING:
    from marscope.schema import VideoResult
    from marscope.video.preprocess import VideoMeta


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="marscope",
        description="See what a video deepfake detector sees. An educational tool: scores are model outputs.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="analyze one video file")
    source.add_argument("--serve", action="store_true", help="run the local UI and REST API")
    parser.add_argument("--models", default=None, help="mars (default), gend, dfd-hr, comma-separated, or all")
    parser.add_argument("--sequential", action="store_true", help="run one model at a time to fit a small GPU")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "fp32"],
        default="auto",
        help="model file on the GPU: auto is fp16 (default); the CPU always runs fp32",
    )
    parser.add_argument("--jurisdiction", choices=JURISDICTIONS, help="region for context notes")
    parser.add_argument("--language", choices=["en", "fr"], default="en")
    parser.add_argument("--no-faces", action="store_true", help="skip face detection (A1 and the A3 face checks)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print the full result as JSON")
    output.add_argument("--report", choices=["markdown", "html", "pdf"], help="write a report next to the video")
    parser.add_argument("--no-llm", action="store_true", help="use the template report even if an LLM is set up")
    parser.add_argument(
        "--env-file", type=Path, help="settings file with the MARSCOPE_LLM_* values (default: .env in this folder)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (defaults to ``sys.argv``).

    Returns
    -------
    int
        Exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if args.env_file is not None:
        if not args.env_file.is_file():
            print(f"marscope: {args.env_file} does not exist", file=sys.stderr)
            return 2
        os.environ["MARSCOPE_ENV_FILE"] = str(args.env_file.resolve())
    from marscope.pipeline import MARScope  # noqa: PLC0415 - keep --help fast

    try:
        models = parse_models(args.models)
    except ValueError as error:
        print(f"marscope: {error}", file=sys.stderr)
        return 2
    toolkit = MARScope(
        models=models,
        device=args.device,
        sequential=args.sequential,
        faces=not args.no_faces,
        precision=args.precision,
    )

    if args.serve:
        from marscope.api import serve  # noqa: PLC0415

        preload = models if args.models != "all" else ["mars"]  # with all, the others load when picked
        print(serve_banner(args.host, args.port))
        serve(host=args.host, port=args.port, toolkit=toolkit, preload=preload)
        return 0

    if not args.video.exists():
        print(f"marscope: {args.video} does not exist", file=sys.stderr)
        return 2

    def progress(event: dict[str, object]) -> None:
        if args.json:
            return
        if event.get("type") == "stage" and event.get("stage") == "loading":
            print(f"loading {event['model']}…", file=sys.stderr)
        elif event.get("type") == "clip":
            print(f"\r{event['model']}: clip {event['done']}", end="", file=sys.stderr)

    import av  # noqa: PLC0415

    from marscope.video.preprocess import describe_read_error, read_meta  # noqa: PLC0415

    try:
        for warning in size_warnings(read_meta(args.video), len(models)):
            print(f"marscope: {warning}", file=sys.stderr)
        result = toolkit.analyze(args.video, jurisdiction=args.jurisdiction, language=args.language, progress=progress)
    except (ValueError, av.FFmpegError) as error:
        print(f"marscope: could not read {args.video.name}: {describe_read_error(error)}", file=sys.stderr)
        return 2
    if not args.json:
        print(file=sys.stderr)
    if args.json:
        print(result.model_dump_json(indent=2))
    elif args.report:
        _write_report(args, result)
    else:
        print(format_summary(result.model_dump(mode="json")))
    return 0


def _write_report(args: argparse.Namespace, result: "VideoResult") -> None:
    """Write the report next to the video, in the format asked for."""
    from marscope.explain.report import build_report, to_html, to_markdown, to_pdf  # noqa: PLC0415

    report = build_report(result, use_llm=not args.no_llm, language=args.language)
    if not args.no_llm and report.summary_source == "template":
        print(f"marscope: {report.by_line}", file=sys.stderr)
    suffix = {"markdown": ".md", "html": ".html", "pdf": ".pdf"}[args.report]
    out = args.video.with_name(f"{args.video.stem}-report{suffix}")
    if args.report == "pdf":
        out.write_bytes(to_pdf(report))
    else:
        out.write_text(to_markdown(report) if args.report == "markdown" else to_html(report), encoding="utf-8")
    print(f"report written to {out}")


def size_warnings(meta: "VideoMeta", n_models: int) -> list[str]:
    """Warn when a video is past the web app's limits; the CLI still analyzes it.

    Parameters
    ----------
    meta : VideoMeta
        From :func:`marscope.video.preprocess.read_meta`.
    n_models : int
        Models that will run.

    Returns
    -------
    list of str
        Warnings (empty for a video the web app would take).
    """
    from marscope.video.preprocess import MAX_LONG_SIDE, MAX_SECONDS, MAX_SHORT_SIDE  # noqa: PLC0415

    out = []
    long_side, short_side = max(meta.width, meta.height), min(meta.width, meta.height)
    if meta.duration_s > MAX_SECONDS:
        clips = math.ceil(meta.duration_s * (meta.fps or 30) / 2 / 64)
        out.append(
            f"the video is {meta.duration_s:.0f} s, longer than the web app's {MAX_SECONDS:.0f} s limit: about "
            f"{clips} clips for each of {n_models} model(s), so this can take many minutes"
        )
    if long_side > MAX_LONG_SIDE or short_side > MAX_SHORT_SIDE:
        out.append(f"the video is {meta.width}x{meta.height}, above 1080p: decoding is slower and uses more memory")
    return out


def serve_banner(host: str, port: int) -> str:
    """Return the ``--serve`` start message: the link, and how to reach it remotely.

    Parameters
    ----------
    host : str
        Interface the server listens on.
    port : int
        Port.

    Returns
    -------
    str
        One or two lines.

    Examples
    --------
    >>> serve_banner("127.0.0.1", 8000)
    'MARScope at http://localhost:8000/app  (the video stays on this machine)'
    """
    local = host in ("127.0.0.1", "localhost", "0.0.0.0", "::")  # noqa: S104 - only compared, never bound here
    lines = [f"MARScope at http://{'localhost' if local else host}:{port}/app  (the video stays on this machine)"]
    if host in ("0.0.0.0", "::"):  # noqa: S104
        name = socket.gethostname()
        lines.append(
            f"Listening on all interfaces. From another machine, open http://{name}:{port}/app or forward a port: "
            f"ssh -N -L 8010:{name}:{port} <user>@<login-host>, then http://localhost:8010/app"
        )
    return "\n".join(lines)


def format_summary(result: dict[str, object]) -> str:
    """Plain-text summary for the terminal.

    Parameters
    ----------
    result : dict
        ``VideoResult.model_dump(mode="json")``.

    Returns
    -------
    str
        Scores in context and the notes.
    """
    video = result["video"]
    assert isinstance(video, dict)
    lines = [
        f"{video['name']}  {video['duration_s']:.1f} s  {video['width']}x{video['height']}  {video['n_clips']} clips",
        "",
    ]
    reading = result.get("reading")
    if isinstance(reading, dict):
        lines += [str(reading["headline"]), *reading["reasons"], str(reading["action"]), ""]
    models = result["models"]
    assert isinstance(models, dict)
    for score in models.values():
        ctx = score["context"]
        label, odds = rarity(ctx["real_rank"], score["name"])
        low, high = (round(v * 100) for v in ctx["real_range"])
        lines.append(
            f"{score['name']:<7} probability of fake {round(score['fake'] * 100)}%  "
            f"[{score['device']} {score['precision']}]",
        )
        lines.append(f"        {label}: {odds} (real videos usually {low}–{high}%)")
        if score.get("device_note"):
            lines.append(f"        note: {score['device_note']}")
    lines.append("")
    notes = result["notes"]
    assert isinstance(notes, list)
    for note in notes:
        prefix = {"at_risk": "!", "breaks": "!", "partly": "~", "holds": "-", "info": " "}[note["state"]]
        lines.append(f"{prefix} {note['text']}")
    lines += ["", "Scores are model outputs and can be wrong. Use --json for every value and detail."]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
