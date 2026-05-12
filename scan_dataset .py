"""
Scan Dataset
Loads pre-cropped word/glyph images from disk and produces tensors
compatible with the visual_embeddings_torch pipeline.

Manifest format (manifest.json):
    {
        "name":     "coptic_papyrus",
        "script":   "coptic",
        "language": null,
        "source":   "British Museum MS 1234",
        "images":   ["images/0001.png", "images/0002.png"]
    }

Directory layout:
    data/
      my_script/
        manifest.json
        images/
          0001.png
          0002.png
          ...

Output tensors: (1, H, W) float32, 0=black 1=white — identical to word_to_tensor().
"""

import os
import json
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Must match visual_embeddings_torch.py
IMG_W = 384
IMG_H = 96


# ─── MANIFEST ────────────────────────────────────────────────────────────────

def load_manifest(manifest_path):
    """
    Load a manifest.json. Returns dict with resolved absolute image paths.
    Raises if the manifest is missing required fields or images are not found.
    """
    manifest_path = os.path.abspath(manifest_path)
    base_dir = os.path.dirname(manifest_path)

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    required = ("name", "script", "images")
    for field in required:
        if field not in data:
            raise ValueError(f"manifest missing field: '{field}' in {manifest_path}")

    if not data["images"]:
        raise ValueError(f"manifest has no images: {manifest_path}")

    resolved = []
    for rel_path in data["images"]:
        abs_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"image not found: {abs_path}")
        resolved.append(abs_path)

    return {
        "name":     data["name"],
        "script":   data["script"],
        "language": data.get("language"),
        "source":   data.get("source"),
        "images":   resolved,
    }


# ─── IMAGE → TENSOR ──────────────────────────────────────────────────────────

def scan_to_tensor(image_path):
    """
    Load a cropped word image and convert to tensor.

    - Converts to grayscale
    - Resizes to IMG_W x IMG_H (128x32) with LANCZOS
    - Normalises to float32 in [0, 1], 0=black 1=white

    Returns: (1, H, W) float32 tensor — same shape and range as word_to_tensor().
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


# ─── DATASET ─────────────────────────────────────────────────────────────────

class ScanPairDataset(Dataset):
    """
    Builds contrastive pairs from one or more scan manifests.

    Positive pairs (label=1):
        Two images from the same script.

    Negative pairs (label=0):
        Two images from different scripts.
        If only one script is loaded, negatives are random pairs from
        different images within that script (label=0 by convention —
        they are not positives and should be pushed apart).

    Args:
        manifest_paths: list of paths to manifest.json files
        n_pairs:        total number of pairs to build
        pos_ratio:      fraction of pairs that are positive (default 0.5)
    """

    def __init__(self, manifest_paths, n_pairs=10000, pos_ratio=0.5):
        self.pairs = []

        manifests = [load_manifest(p) for p in manifest_paths]

        # Group images by script
        by_script = {}
        for m in manifests:
            script = m["script"]
            by_script.setdefault(script, []).extend(m["images"])

        scripts = list(by_script.keys())
        n_pos = int(n_pairs * pos_ratio)
        n_neg = n_pairs - n_pos

        print(f"ScanPairDataset: {len(scripts)} script(s), {n_pairs} pairs")
        for s, imgs in by_script.items():
            print(f"  {s}: {len(imgs)} images")

        # Positive pairs
        for _ in range(n_pos):
            script = random.choice(scripts)
            imgs = by_script[script]
            if len(imgs) < 2:
                a = b = imgs[0]
            else:
                a, b = random.sample(imgs, 2)
            self.pairs.append((a, b, 1.0))

        # Negative pairs
        for _ in range(n_neg):
            if len(scripts) >= 2:
                s1, s2 = random.sample(scripts, 2)
                a = random.choice(by_script[s1])
                b = random.choice(by_script[s2])
            else:
                # Single script: pick two different images
                imgs = by_script[scripts[0]]
                if len(imgs) < 2:
                    a = b = imgs[0]
                else:
                    a, b = random.sample(imgs, 2)
            self.pairs.append((a, b, 0.0))

        random.shuffle(self.pairs)

        # Pre-load all tensors into RAM
        print(f"  Loading {len(self.pairs)} image pairs into RAM...")
        self._tensors = []
        seen = {}
        for path_a, path_b, label in self.pairs:
            if path_a not in seen:
                seen[path_a] = scan_to_tensor(path_a)
            if path_b not in seen:
                seen[path_b] = scan_to_tensor(path_b)
            self._tensors.append((
                seen[path_a],
                seen[path_b],
                torch.tensor(label, dtype=torch.float32),
            ))
        print("  Done.")

    def __len__(self):
        return len(self._tensors)

    def __getitem__(self, idx):
        return self._tensors[idx]


# ─── SMOKE TEST ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scan_dataset.py path/to/manifest.json [path/to/manifest2.json ...]")
        print()
        print("Smoke test — checking tensor shape and value range.")
        sys.exit(0)

    manifests = sys.argv[1:]
    ds = ScanPairDataset(manifests, n_pairs=20)

    ta, tb, label = ds[0]
    print(f"\nFirst pair:")
    print(f"  ta shape:  {tuple(ta.shape)}  (expected: (1, {IMG_H}, {IMG_W}))")
    print(f"  tb shape:  {tuple(tb.shape)}")
    print(f"  label:     {label.item()}")
    print(f"  ta range:  [{ta.min():.3f}, {ta.max():.3f}]  (expected: [0, 1])")

    shape_ok = ta.shape == (1, IMG_H, IMG_W)
    range_ok = 0.0 <= ta.min() and ta.max() <= 1.0
    print(f"\n  Shape: {'OK' if shape_ok else 'FAIL'}")
    print(f"  Range: {'OK' if range_ok else 'FAIL'}")
