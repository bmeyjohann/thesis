"""Validate rendered target-terrain artifacts and report coverage."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from PIL import Image


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "visualizations" / "unitree_target_terrain_variants"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    variants = manifest["variants"]
    if len(variants) != 6:
        raise RuntimeError(f"Expected six variants, found {len(variants)}")
    for item in variants:
        for key in ("topdown", "perspective", "video", "metadata"):
            path = Path(item[key])
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"Missing or empty {key}: {path}")
        for key in ("topdown", "perspective"):
            with Image.open(item[key]) as image:
                image.verify()
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,duration",
                "-of", "json", item["video"],
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        if stream["codec_name"] != "h264":
            raise RuntimeError(f"Unexpected codec for {item['video']}: {stream}")

    report = {
        "variants": len(variants),
        "decoded_images": 2 * len(variants) + 1,
        "decoded_videos": len(variants),
        "coverage": [
            {"preset": item["preset"], "seed": item["seed"], **item["coverage"]}
            for item in variants
        ],
    }
    (root / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

