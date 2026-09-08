# LayerUnfold Decomposition Core

A small, deterministic Python library for splitting an opaque 8-bit RGB image
into layers that reconstruct the source pixels exactly.

The library takes two same-size images:

- **original** — the image whose pixels must be preserved
- **base guide** — a lighting-free or otherwise editable base supplied by the caller

It does not generate the base guide. It only computes the correction layers and
verifies that integer recomposition returns every original RGB channel exactly.

## What is included

- `Normal -> Multiply -> Linear Dodge (Add)` decomposition
- `Normal -> Multiply -> Screen` decomposition
- Base plus a compact RGBA Normal correction layer
- A CLI that writes PNG layers and a machine-readable verification report
- Randomized exact-recomposition tests and license-safe synthetic examples

## Scope

This repository contains only the standalone numerical decomposition library.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## CLI

```bash
layerunfold-decompose original.png base-guide.png \
  --output decomposed \
  --mode multiply-add
```

Modes:

| Mode | Output stack, bottom to top |
|---|---|
| `multiply-add` | Base (Normal), Shadow (Multiply), Highlight (Linear Dodge/Add) |
| `multiply-screen` | Base (Normal), Shadow (Multiply), Highlight (Screen) |
| `normal-overlay` | Base (Normal), Detail (Normal RGBA) |

The output directory also contains `recomposed.png` and `report.json`. A successful
run reports `"max_reconstruction_error_u8": 0`.

Only fully opaque RGB/RGBA inputs are currently accepted. Transparent, 16-bit,
32-bit, CMYK, Lab, HDR, and editor-specific blend semantics are intentionally
rejected or out of scope rather than silently approximated.

## Python API

```python
import numpy as np
from layerunfold_decompose import (
    decompose_multiply_add_exact,
    recompose_multiply_add,
)

original: np.ndarray = ...  # uint8, HxWx3
base: np.ndarray = ...  # uint8, HxWx3

base_out, shadow, highlight = decompose_multiply_add_exact(original, base)
reconstructed = recompose_multiply_add(base_out, shadow, highlight)
assert np.array_equal(reconstructed, original)
```

## Numerical contract

For each 8-bit channel, the Multiply layer is chosen so the rounded integer
product never exceeds the original channel. The remaining positive difference is
represented by the selected highlight blend. Every public decomposition function
runs its matching recomposition function and fails closed if the result differs.

Pixel-exact reconstruction does not mean the supplied base guide is semantically
correct or pleasant to edit. Those are separate concerns owned by the caller.

## Development

```bash
python -m pytest
ruff check .
ruff format --check .
python examples/make_synthetic_example.py
```

## License

MIT. See [LICENSE](LICENSE).
