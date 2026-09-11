"""Generate synthetic placeholder sample images for the sample gallery.

Purpose:
    Populate `assets/samples/` with a handful of small, deterministic,
    synthetic images so the Image Acquisition sample gallery has
    something to show without requiring a real leaf-disease dataset
    (which isn't available in this environment / isn't licensed for
    redistribution).

Description:
    These are clearly NOT real leaf photographs — they're simple
    procedural shapes (a green blob with brown "spot" markings) that
    stand in for "a leaf image" just well enough to exercise upload,
    preview, metadata, and resizing logic end to end. Replace the
    contents of `assets/samples/` with real, appropriately-licensed
    leaf images before using this project for actual disease
    classification work (Milestones 6-7).

Dependencies:
    Pillow, NumPy.

Public functions:
    main() -> None
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "samples"

_LEAF_GREEN = (86, 140, 74)
_SPOT_BROWN = (120, 72, 40)
_BACKGROUND = (235, 232, 220)


def _make_synthetic_leaf(seed: int, size: tuple[int, int] = (512, 384)) -> Image.Image:
    """Draw one procedural leaf-like placeholder image.

    Args:
        seed: Random seed, so output is deterministic and reproducible.
        size: (width, height) of the generated image in pixels.

    Returns:
        An RGB PIL Image — a green ellipse "leaf" with a visible
        midrib line and a random scatter of brown "disease spot" ellipses.
    """
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", size, _BACKGROUND)
    draw = ImageDraw.Draw(image)

    width, height = size
    margin_x, margin_y = int(width * 0.12), int(height * 0.08)
    draw.ellipse(
        [margin_x, margin_y, width - margin_x, height - margin_y],
        fill=_LEAF_GREEN,
    )
    draw.line(
        [(width // 2, margin_y), (width // 2, height - margin_y)],
        fill=(60, 100, 52),
        width=3,
    )

    spot_count = int(rng.integers(3, 9))
    for _ in range(spot_count):
        cx = rng.integers(margin_x + 20, width - margin_x - 20)
        cy = rng.integers(margin_y + 20, height - margin_y - 20)
        r = rng.integers(6, 22)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_SPOT_BROWN)

    return image


def main() -> None:
    """Generate and save the synthetic sample gallery images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(1, 5):
        image = _make_synthetic_leaf(seed=index)
        out_path = OUTPUT_DIR / f"sample_leaf_{index:02d}.png"
        image.save(out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
