"""Public API for the decomposition core."""

from .core import (
    decompose_multiply_add_exact,
    decompose_multiply_screen_exact,
    decompose_normal_overlay_exact,
    recompose_multiply_add,
    recompose_multiply_screen,
    recompose_normal,
)

__all__ = [
    "decompose_multiply_add_exact",
    "decompose_multiply_screen_exact",
    "decompose_normal_overlay_exact",
    "recompose_multiply_add",
    "recompose_multiply_screen",
    "recompose_normal",
]

__version__ = "0.1.0"
