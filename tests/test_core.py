from __future__ import annotations

import numpy as np
import pytest

from layerunfold_decompose import (
    decompose_multiply_add_exact,
    decompose_multiply_screen_exact,
    decompose_normal_overlay_exact,
    recompose_multiply_add,
    recompose_multiply_screen,
    recompose_normal,
)


@pytest.mark.parametrize("seed", [0, 1, 42, 1729])
def test_multiply_add_is_pixel_exact(seed: int) -> None:
    rng = np.random.default_rng(seed)
    original = rng.integers(0, 256, size=(97, 131, 3), dtype=np.uint8)
    base = rng.integers(0, 256, size=(97, 131, 3), dtype=np.uint8)
    base_out, shadow, highlight = decompose_multiply_add_exact(original, base)
    assert np.array_equal(recompose_multiply_add(base_out, shadow, highlight), original)


@pytest.mark.parametrize("seed", [0, 1, 42, 1729])
def test_multiply_screen_is_pixel_exact(seed: int) -> None:
    rng = np.random.default_rng(seed)
    original = rng.integers(0, 256, size=(97, 131, 3), dtype=np.uint8)
    base = rng.integers(0, 256, size=(97, 131, 3), dtype=np.uint8)
    base_out, shadow, highlight = decompose_multiply_screen_exact(original, base)
    assert np.array_equal(recompose_multiply_screen(base_out, shadow, highlight), original)


@pytest.mark.parametrize("seed", [0, 1, 42, 1729])
def test_normal_overlay_is_pixel_exact(seed: int) -> None:
    rng = np.random.default_rng(seed)
    original = rng.integers(0, 256, size=(97, 131, 3), dtype=np.uint8)
    base = rng.integers(0, 256, size=(97, 131, 3), dtype=np.uint8)
    base_out, detail = decompose_normal_overlay_exact(original, base)
    assert np.array_equal(recompose_normal(base_out, detail), original)


def test_equal_pixels_need_no_normal_overlay() -> None:
    pixels = np.full((8, 9, 3), (12, 34, 56), dtype=np.uint8)
    _, detail = decompose_normal_overlay_exact(pixels, pixels)
    assert np.count_nonzero(detail[..., 3]) == 0


def test_thin_dark_line_does_not_spread() -> None:
    base = np.full((65, 65, 3), 204, dtype=np.uint8)
    original = base.copy()
    original[:, 32] = (26, 40, 52)
    _, shadow, highlight = decompose_multiply_add_exact(original, base)
    assert np.all(shadow[:, 31] == 255)
    assert np.all(shadow[:, 33] == 255)
    assert np.all(shadow[:, 32] < 255)
    assert np.array_equal(recompose_multiply_add(base, shadow, highlight), original)


def test_colored_shadow_channels_are_preserved() -> None:
    base = np.array([[[200, 200, 200]]], dtype=np.uint8)
    original = np.array([[[100, 150, 200]]], dtype=np.uint8)
    _, shadow, _ = decompose_multiply_add_exact(original, base)
    assert len(set(int(value) for value in shadow[0, 0])) == 3


def test_invalid_shape_and_range_fail_closed() -> None:
    with pytest.raises(ValueError, match="same-size"):
        decompose_multiply_add_exact(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((3, 2, 3), dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="0..255"):
        decompose_multiply_add_exact(
            np.zeros((1, 1, 3), dtype=np.int16),
            np.full((1, 1, 3), 256, dtype=np.int16),
        )
    with pytest.raises(TypeError, match="integer"):
        decompose_multiply_add_exact(
            np.zeros((1, 1, 3), dtype=np.float32),
            np.zeros((1, 1, 3), dtype=np.float32),
        )
