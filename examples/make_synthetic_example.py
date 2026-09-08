"""Create a license-safe synthetic input pair and decompose it."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from layerunfold_decompose.cli import main


def make_example(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    base_image = Image.new("RGB", (512, 320), (238, 231, 215))
    draw = ImageDraw.Draw(base_image)
    draw.ellipse((80, 48, 304, 272), fill=(188, 101, 76))
    draw.rectangle((286, 86, 454, 252), fill=(66, 131, 168))

    base = np.asarray(base_image, dtype=np.uint8).copy()
    original = base.astype(np.float32)
    yy, xx = np.mgrid[:320, :512]
    light = np.clip(0.55 + 0.65 * (xx / 511) + 0.18 * (1 - yy / 319), 0.35, 1.25)
    original = np.clip(original * light[..., None], 0, 255).astype(np.uint8)
    original[145:151, 72:462] = (35, 38, 44)

    original_path = root / "original.png"
    base_path = root / "base-guide.png"
    Image.fromarray(original, mode="RGB").save(original_path)
    Image.fromarray(base, mode="RGB").save(base_path)
    main([str(original_path), str(base_path), "-o", str(root / "decomposed")])


if __name__ == "__main__":
    make_example(Path("example-output"))
