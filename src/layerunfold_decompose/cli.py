"""Command-line interface for the standalone decomposition core."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .core import (
    decompose_multiply_add_exact,
    decompose_multiply_screen_exact,
    decompose_normal_overlay_exact,
    recompose_multiply_add,
    recompose_multiply_screen,
    recompose_normal,
)


def _load_opaque_rgb(path: Path, *, label: str) -> tuple[np.ndarray, bytes | None]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
        if np.any(alpha != 255):
            raise ValueError(f"{label} must be fully opaque; alpha workflows are not supported")
        return np.asarray(rgba.convert("RGB"), dtype=np.uint8), image.info.get("icc_profile")


def _save_rgb(path: Path, pixels: np.ndarray, icc_profile: bytes | None) -> None:
    kwargs = {"icc_profile": icc_profile} if icc_profile else {}
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG", **kwargs)


def _save_rgba(path: Path, pixels: np.ndarray, icc_profile: bytes | None) -> None:
    kwargs = {"icc_profile": icc_profile} if icc_profile else {}
    Image.fromarray(pixels, mode="RGBA").save(path, format="PNG", **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layerunfold-decompose",
        description="Pixel-exact 8-bit RGB image decomposition using a supplied base guide.",
    )
    parser.add_argument("original", type=Path, help="original opaque RGB/RGBA PNG")
    parser.add_argument("base", type=Path, help="same-size opaque base-guide PNG")
    parser.add_argument("--output", "-o", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--mode",
        choices=("multiply-add", "multiply-screen", "normal-overlay"),
        default="multiply-add",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    original, original_icc = _load_opaque_rgb(args.original, label="original")
    base, _ = _load_opaque_rgb(args.base, label="base")
    if original.shape != base.shape:
        raise ValueError("original and base images must have identical dimensions")

    if args.mode == "multiply-add":
        base_out, shadow, highlight = decompose_multiply_add_exact(original, base)
        recomposed = recompose_multiply_add(base_out, shadow, highlight)
        _save_rgb(args.output / "base.png", base_out, original_icc)
        _save_rgb(args.output / "shadow.png", shadow, original_icc)
        _save_rgb(args.output / "highlight.png", highlight, original_icc)
        outputs = ["base.png", "shadow.png", "highlight.png", "recomposed.png"]
        blend_order = ["Normal", "Multiply", "Linear Dodge (Add)"]
    elif args.mode == "multiply-screen":
        base_out, shadow, highlight = decompose_multiply_screen_exact(original, base)
        recomposed = recompose_multiply_screen(base_out, shadow, highlight)
        _save_rgb(args.output / "base.png", base_out, original_icc)
        _save_rgb(args.output / "shadow.png", shadow, original_icc)
        _save_rgb(args.output / "highlight.png", highlight, original_icc)
        outputs = ["base.png", "shadow.png", "highlight.png", "recomposed.png"]
        blend_order = ["Normal", "Multiply", "Screen"]
    else:
        base_out, detail = decompose_normal_overlay_exact(original, base)
        recomposed = recompose_normal(base_out, detail)
        _save_rgb(args.output / "base.png", base_out, original_icc)
        _save_rgba(args.output / "detail.png", detail, original_icc)
        outputs = ["base.png", "detail.png", "recomposed.png"]
        blend_order = ["Normal", "Normal RGBA overlay"]

    _save_rgb(args.output / "recomposed.png", recomposed, original_icc)
    max_error = int(np.max(np.abs(recomposed.astype(np.int16) - original.astype(np.int16))))
    report: dict[str, object] = {
        "mode": args.mode,
        "width": int(original.shape[1]),
        "height": int(original.shape[0]),
        "max_reconstruction_error_u8": max_error,
        "pixel_exact": max_error == 0,
        "blend_order": blend_order,
        "outputs": outputs,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False))
    return 0
