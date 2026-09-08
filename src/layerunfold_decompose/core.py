"""Deterministic, pixel-exact decomposition for 8-bit RGB images.

This module operates only on caller-supplied pixel arrays.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

RGB8 = NDArray[np.uint8]
RGBA8 = NDArray[np.uint8]

_ROW_CHUNK = 256


def _validate_rgb_pair(original_rgb: np.ndarray, base_rgb: np.ndarray) -> tuple[RGB8, RGB8]:
    original_input = np.asarray(original_rgb)
    base_input = np.asarray(base_rgb)
    if (
        original_input.shape != base_input.shape
        or original_input.ndim != 3
        or original_input.shape[2] != 3
    ):
        raise ValueError("original_rgb and base_rgb must be same-size HxWx3 arrays")
    for name, value in (("original_rgb", original_input), ("base_rgb", base_input)):
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"{name} must contain integer channel values")
        if value.size and (np.min(value) < 0 or np.max(value) > 255):
            raise ValueError(f"{name} channel values must be in the range 0..255")
    return original_input.astype(np.uint8, copy=False), base_input.astype(np.uint8, copy=False)


def _validate_three_rgb(
    base_rgb: np.ndarray,
    shadow_rgb: np.ndarray,
    highlight_rgb: np.ndarray,
) -> tuple[RGB8, RGB8, RGB8]:
    base, shadow = _validate_rgb_pair(base_rgb, shadow_rgb)
    _, highlight = _validate_rgb_pair(base, highlight_rgb)
    return base, shadow, highlight


@lru_cache(maxsize=1)
def _screen_highlight_lut() -> NDArray[np.uint8]:
    """Map an 8-bit Screen result back to its exact highlight channel."""
    lookup = np.zeros((256, 256), dtype=np.uint8)
    highlights = np.arange(256, dtype=np.uint16)
    for multiplied in range(256):
        outputs = 255 - (((255 - multiplied) * (255 - highlights) + 127) // 255)
        lookup[multiplied, outputs] = highlights.astype(np.uint8)
        targets = np.arange(multiplied, 256, dtype=np.uint16)
        selected = lookup[multiplied, targets].astype(np.uint16)
        verified = 255 - (((255 - multiplied) * (255 - selected) + 127) // 255)
        if not np.array_equal(verified, targets):
            raise RuntimeError("8-bit Screen inverse lookup is incomplete")
    lookup.setflags(write=False)
    return lookup


def recompose_multiply_add(
    base_rgb: np.ndarray,
    shadow_rgb: np.ndarray,
    highlight_rgb: np.ndarray,
) -> RGB8:
    """Recompose Normal -> Multiply -> Linear Dodge (Add) in 8-bit integer RGB."""
    base, shadow, highlight = _validate_three_rgb(base_rgb, shadow_rgb, highlight_rgb)
    multiplied = (base.astype(np.uint16) * shadow.astype(np.uint16) + 127) // 255
    return np.minimum(multiplied + highlight.astype(np.uint16), 255).astype(np.uint8)


def decompose_multiply_add_exact(
    original_rgb: np.ndarray,
    base_rgb: np.ndarray,
) -> tuple[RGB8, RGB8, RGB8]:
    """Create an exact Base / Shadow / Highlight decomposition.

    The three outputs are intended for Normal, Multiply, and Linear Dodge (Add)
    blending respectively. The input base is a guide, not a prediction made here.
    """
    original, base = _validate_rgb_pair(original_rgb, base_rgb)
    shadow = np.empty_like(base)
    highlight = np.empty_like(base)
    for y0 in range(0, base.shape[0], _ROW_CHUNK):
        y1 = min(y0 + _ROW_CHUNK, base.shape[0])
        source = original[y0:y1].astype(np.uint16)
        guide = base[y0:y1].astype(np.uint16)
        divisor = np.maximum(guide, 1)
        multiplier = np.where(source >= guide, 255, (source * 255) // divisor).astype(np.uint16)
        multiplied = (guide * multiplier + 127) // 255
        if np.any(multiplied > source):
            raise RuntimeError("Multiply component exceeded the source pixel")
        shadow[y0:y1] = multiplier.astype(np.uint8)
        highlight[y0:y1] = (source - multiplied).astype(np.uint8)

    base_out = base.copy()
    if not np.array_equal(recompose_multiply_add(base_out, shadow, highlight), original):
        raise RuntimeError("lossless Multiply + Add invariant failed")
    return base_out, shadow, highlight


def recompose_multiply_screen(
    base_rgb: np.ndarray,
    shadow_rgb: np.ndarray,
    highlight_rgb: np.ndarray,
) -> RGB8:
    """Recompose Normal -> Multiply -> Screen in 8-bit integer RGB."""
    base, shadow, highlight = _validate_three_rgb(base_rgb, shadow_rgb, highlight_rgb)
    base16 = base.astype(np.uint16)
    shadow16 = shadow.astype(np.uint16)
    highlight16 = highlight.astype(np.uint16)
    multiplied = (base16 * shadow16 + 127) // 255
    return (255 - (((255 - multiplied) * (255 - highlight16) + 127) // 255)).astype(np.uint8)


def decompose_multiply_screen_exact(
    original_rgb: np.ndarray,
    base_rgb: np.ndarray,
) -> tuple[RGB8, RGB8, RGB8]:
    """Create an exact Base / Shadow / Highlight decomposition for Screen."""
    original, base = _validate_rgb_pair(original_rgb, base_rgb)
    shadow = np.empty_like(base)
    multiplied = np.empty_like(base)
    for y0 in range(0, base.shape[0], _ROW_CHUNK):
        y1 = min(y0 + _ROW_CHUNK, base.shape[0])
        source = original[y0:y1].astype(np.uint16)
        guide = base[y0:y1].astype(np.uint16)
        divisor = np.maximum(guide, 1)
        multiplier = np.where(source >= guide, 255, (source * 255) // divisor).astype(np.uint16)
        product = (guide * multiplier + 127) // 255
        if np.any(product > source):
            raise RuntimeError("Multiply component exceeded the source pixel")
        shadow[y0:y1] = multiplier.astype(np.uint8)
        multiplied[y0:y1] = product.astype(np.uint8)

    highlight = _screen_highlight_lut()[multiplied, original]
    base_out = base.copy()
    if not np.array_equal(recompose_multiply_screen(base_out, shadow, highlight), original):
        raise RuntimeError("lossless Multiply + Screen invariant failed")
    return base_out, shadow, highlight


def recompose_normal(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> RGB8:
    """Recompose a same-size RGBA overlay over a base with 8-bit Normal blending."""
    base_input = np.asarray(base_rgb)
    overlay_input = np.asarray(overlay_rgba)
    if base_input.ndim != 3 or base_input.shape[2] != 3:
        raise ValueError("base_rgb must be an HxWx3 array")
    if overlay_input.shape != (*base_input.shape[:2], 4):
        raise ValueError("overlay_rgba must be a same-size HxWx4 array")
    for name, value in (("base_rgb", base_input), ("overlay_rgba", overlay_input)):
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"{name} must contain integer channel values")
        if value.size and (np.min(value) < 0 or np.max(value) > 255):
            raise ValueError(f"{name} channel values must be in the range 0..255")
    base = base_input.astype(np.uint16, copy=False)
    overlay = overlay_input.astype(np.uint16, copy=False)
    alpha = overlay[..., 3:4]
    return ((alpha * overlay[..., :3] + (255 - alpha) * base + 127) // 255).astype(np.uint8)


def decompose_normal_overlay_exact(
    original_rgb: np.ndarray,
    base_rgb: np.ndarray,
) -> tuple[RGB8, RGBA8]:
    """Create Base plus a compact RGBA correction for Normal blending.

    Pixels equal to the base are fully transparent. Other pixels use the smallest
    shared integer alpha that can represent all three channels, with an opaque
    pixel fallback if 8-bit quantization makes the analytical solution inexact.
    """
    original, base = _validate_rgb_pair(original_rgb, base_rgb)
    overlay = np.empty((*base.shape[:2], 4), dtype=np.uint8)
    overlay[..., :3] = original
    alpha_out = overlay[..., 3]
    for y0 in range(0, base.shape[0], _ROW_CHUNK):
        y1 = min(y0 + _ROW_CHUNK, base.shape[0])
        source = original[y0:y1].astype(np.uint32)
        guide = base[y0:y1].astype(np.uint32)
        up_denominator = np.maximum(255 - guide, 1)
        down_denominator = np.maximum(guide, 1)
        up_numerator = np.where(source > guide, (source - guide) * 255, 0)
        down_numerator = np.where(source < guide, (guide - source) * 255, 0)
        required_up = (up_numerator + up_denominator - 1) // up_denominator
        required_down = (down_numerator + down_denominator - 1) // down_denominator
        alpha = np.max(np.maximum(required_up, required_down), axis=2).astype(np.uint32)

        nonzero = alpha > 0
        color = source.copy()
        if np.any(nonzero):
            alpha_3d = alpha[..., None]
            numerator = 255 * source - (255 - alpha_3d) * guide
            safe_alpha = np.maximum(alpha_3d, 1)
            color = np.clip((numerator + safe_alpha // 2) // safe_alpha, 0, 255)
            reconstructed = (alpha_3d * color + (255 - alpha_3d) * guide + 127) // 255
            bad = nonzero & np.any(reconstructed != source, axis=2)
            if np.any(bad):
                alpha[bad] = 255
                color[bad] = source[bad]
        overlay[y0:y1, ..., :3] = color.astype(np.uint8)
        alpha_out[y0:y1] = alpha.astype(np.uint8)

    base_out = base.copy()
    if not np.array_equal(recompose_normal(base_out, overlay), original):
        raise RuntimeError("lossless Normal overlay invariant failed")
    return base_out, overlay
