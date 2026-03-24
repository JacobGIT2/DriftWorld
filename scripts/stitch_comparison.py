"""Stitch individual model videos into side-by-side comparison.

Reads per-model videos from outputs/world_model_rollouts/final/ and
concatenates them horizontally into a single comparison video.

Usage:
    # Stitch whatever is available now (skip missing models):
    uv run python scripts/stitch_comparison.py

    # Stitch specific models:
    uv run python scripts/stitch_comparison.py --models gt sim vae flow5 ddpm1000
"""

import argparse

import cv2
import imageio
import numpy as np
from pathlib import Path
from PIL import Image

INPUT_DIR = Path("outputs/world_model_rollouts/final")
CELL_SIZE = 200
FPS = 12


def stitch(direction: str, models: list[str], output_path: Path):
    """Stitch videos for a given direction (forward/reversed)."""
    # Load all available videos
    all_frames = {}
    for model in models:
        video_path = INPUT_DIR / f"{model}_{direction}.mp4"
        if not video_path.exists():
            print(f"  Skipping {model} ({video_path} not found)")
            continue
        frames = imageio.mimread(str(video_path))
        all_frames[model] = frames
        print(f"  Loaded {model}: {len(frames)} frames")

    if not all_frames:
        print(f"  No videos found for {direction}, skipping")
        return

    # Stitch
    max_len = max(len(f) for f in all_frames.values())
    combined = []
    for i in range(max_len):
        panels = []
        for model in all_frames:
            frames = all_frames[model]
            f = frames[min(i, len(frames) - 1)]
            # Resize to uniform cell size (strip existing labels first)
            f = np.array(Image.fromarray(f).resize((CELL_SIZE, CELL_SIZE)))
            panels.append(f)
        combined.append(np.concatenate(panels, axis=1))

    imageio.mimwrite(str(output_path), combined, fps=FPS)

    # Save key frames
    stem = output_path.stem
    parent = output_path.parent
    Image.fromarray(combined[0]).save(str(parent / f"{stem}_first.png"))
    Image.fromarray(combined[len(combined) // 2]).save(str(parent / f"{stem}_mid.png"))
    Image.fromarray(combined[-1]).save(str(parent / f"{stem}_last.png"))

    print(f"  Saved {output_path} ({len(combined)} frames, {len(all_frames)} columns)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gt", "sim", "vae", "flow5", "ddpm1000"],
        help="Models to include (order = left to right columns)",
    )
    args = parser.parse_args()

    output_dir = INPUT_DIR / "stitched"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Forward ===")
    stitch("forward", args.models, output_dir / "forward.mp4")

    print("=== Reversed ===")
    stitch("reversed", args.models, output_dir / "reversed.mp4")

    print("Done!")


if __name__ == "__main__":
    main()
