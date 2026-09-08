from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from layerunfold_decompose.cli import main


@pytest.mark.parametrize("mode", ["multiply-add", "multiply-screen", "normal-overlay"])
def test_cli_writes_verified_outputs(tmp_path: Path, mode: str) -> None:
    rng = np.random.default_rng(42)
    original = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    base = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    original_path = tmp_path / "original.png"
    base_path = tmp_path / "base.png"
    output_path = tmp_path / "result"
    Image.fromarray(original, mode="RGB").save(original_path)
    Image.fromarray(base, mode="RGB").save(base_path)

    assert main([str(original_path), str(base_path), "-o", str(output_path), "--mode", mode]) == 0

    report = json.loads((output_path / "report.json").read_text(encoding="utf-8"))
    assert report["pixel_exact"] is True
    assert report["max_reconstruction_error_u8"] == 0
    recomposed = np.asarray(Image.open(output_path / "recomposed.png").convert("RGB"))
    assert np.array_equal(recomposed, original)


def test_cli_rejects_transparency(tmp_path: Path) -> None:
    original_path = tmp_path / "original.png"
    base_path = tmp_path / "base.png"
    Image.new("RGBA", (2, 2), (1, 2, 3, 128)).save(original_path)
    Image.new("RGB", (2, 2), (1, 2, 3)).save(base_path)
    with pytest.raises(SystemExit, match="fully opaque"):
        main([str(original_path), str(base_path), "-o", str(tmp_path / "result")])
