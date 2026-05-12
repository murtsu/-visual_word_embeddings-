"""
vwe.py — Visual Word Embeddings shell

Add a new script:
    python vwe.py add --name coptic --script coptic --images data/coptic/images/

List available adapters:
    python vwe.py list

Encode an image:
    python vwe.py encode --image data/coptic/images/0001.png

Find nearest neighbours across all loaded scripts:
    python vwe.py neighbours --image data/coptic/images/0001.png
"""

import argparse
import os
import json
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scan_dataset import load_manifest, scan_to_tensor, IMG_W, IMG_H

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_MODEL   = "visual_embeddings.pt"
ADAPTER_DIR  = "adapters"


def _is_cjk(word):
    for ch in word:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7A3:
            return True
    return False


# ─── MODEL ───────────────────────────────────────────────────────────────────

class VisualWordEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.projection(self.cnn(x)), p=2, dim=1)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = VisualWordEncoder(embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def encode_image(model, image_path):
    t = scan_to_tensor(image_path).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(t)
    return emb.squeeze(0).cpu().numpy()


def adapter_path(name):
    return os.path.join(ADAPTER_DIR, f"{name}.pt")


def manifest_path(name):
    return os.path.join(ADAPTER_DIR, f"{name}_manifest.json")


# ─── COMMANDS ────────────────────────────────────────────────────────────────

def cmd_add(args):
    """Create manifest and fine-tune on new script images."""
    import subprocess, sys

    images_dir = os.path.abspath(args.images)
    if not os.path.isdir(images_dir):
        print(f"ERROR: not a directory: {images_dir}")
        return

    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files.sort()

    if not image_files:
        print(f"ERROR: no images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images in {images_dir}")

    # Write manifest
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    mpath = manifest_path(args.name)
    manifest = {
        "name":     args.name,
        "script":   args.script,
        "language": args.language,
        "source":   args.source,
        "images":   image_files,
    }
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Manifest written: {mpath}")

    # Fine-tune
    out = adapter_path(args.name)
    cmd = [
        sys.executable, "finetune_scan.py",
        "--base",      args.base,
        "--manifests", mpath,
        "--output",    out,
        "--epochs",    str(args.epochs),
        "--pairs",     str(args.pairs),
        "--batch",     str(args.batch),
        "--lr",        str(args.lr),
        "--unfreeze",  str(args.unfreeze),
        "--vocab",     args.vocab,
    ]
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"\nAdapter saved: {out}")


def cmd_list(args):
    """List available adapters."""
    if not os.path.isdir(ADAPTER_DIR):
        print("No adapters yet. Run: python vwe.py add ...")
        return

    pts = glob.glob(os.path.join(ADAPTER_DIR, "*.pt"))
    if not pts:
        print("No adapters yet.")
        return

    print(f"{'NAME':<20} {'TIER':<10} {'SCRIPT':<20} IMAGES")
    print("-" * 65)
    for pt in sorted(pts):
        name = os.path.splitext(os.path.basename(pt))[0]
        mpath = manifest_path(name)
        # Read tier from checkpoint if available
        try:
            ckpt = torch.load(pt, map_location="cpu")
            tier = ckpt.get("tier", "adapter")
            langs = ckpt.get("langs", None)
        except Exception:
            tier = "?"
            langs = None
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                m = json.load(f)
            script   = m.get("script", "?")
            n_images = len(m.get("images", []))
        elif langs:
            script   = ",".join(langs)
            n_images = "-"
        else:
            script   = "?"
            n_images = "?"
        print(f"{name:<20} {tier:<10} {script:<20} {n_images}")


def cmd_encode(args):
    """Encode a single image and print the embedding vector."""
    if args.adapter:
        model = load_model(adapter_path(args.adapter))
        print(f"Model: {adapter_path(args.adapter)}")
    else:
        model = load_model(args.base)
        print(f"Model: {args.base}")

    emb = encode_image(model, args.image)
    print(f"Image: {args.image}")
    print(f"Shape: {emb.shape}")
    print(f"Norm:  {np.linalg.norm(emb):.4f}")
    print(f"First 8 values: {emb[:8]}")


def cmd_neighbours(args):
    """Find nearest neighbours for a query image across all images in an adapter."""
    if not args.adapter:
        print("ERROR: --adapter required for neighbours")
        return

    model = load_model(adapter_path(args.adapter))
    mpath = manifest_path(args.adapter)
    if not os.path.exists(mpath):
        print(f"ERROR: manifest not found: {mpath}")
        return

    with open(mpath, encoding="utf-8") as f:
        m = json.load(f)

    query_emb = encode_image(model, args.image)

    scored = []
    for img_path in m["images"]:
        if os.path.abspath(img_path) == os.path.abspath(args.image):
            continue
        emb = encode_image(model, img_path)
        scored.append((cosine_sim(query_emb, emb), img_path))

    scored.sort(reverse=True)
    print(f"Query: {args.image}")
    print(f"Top {args.n} neighbours:")
    for sim, path in scored[:args.n]:
        print(f"  {sim:+.4f}  {path}")


# ─── SEARCH ──────────────────────────────────────────────────────────────────

def cmd_search(args):
    """
    Encode a scan image and find nearest neighbours in the base vocabulary.
    Uses the adapter model (same embedding space as base) to encode both
    the query image and rendered vocabulary words.
    """
    from PIL import Image, ImageDraw, ImageFont
    import subprocess

    model = load_model(adapter_path(args.adapter) if args.adapter else args.base)

    # Render a word string to tensor — mirrors word_to_tensor() in the training code
    def word_to_tensor(word, font_path, font_size=20):
        if _is_cjk(word):
            font_size = 28
        font = ImageFont.truetype(font_path, font_size)
        img  = Image.new("L", (IMG_W, IMG_H), color=255)
        draw = ImageDraw.Draw(img)
        try:
            bbox = draw.textbbox((0, 0), word, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            w, h = int(font.getlength(word)), font_size
        x = max(2, (IMG_W - w) // 2)
        y = max(2, (IMG_H - h) // 2)
        draw.text((x, y), word, fill=0, font=font)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    # Find a usable font
    def find_font():
        candidates = [
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        try:
            r = subprocess.run(["fc-match", "-f", "%{file}", "FreeSans"],
                               capture_output=True, text=True, timeout=3)
            if r.returncode == 0 and os.path.exists(r.stdout.strip()):
                return r.stdout.strip()
        except Exception:
            pass
        raise RuntimeError("No font found. Install fonts-freefont-ttf.")

    font_path = find_font()

    # Load vocabulary words
    vocab_words = []
    vocab_dir   = args.vocab

    if os.path.isdir(vocab_dir):
        for fname in sorted(os.listdir(vocab_dir)):
            if not fname.endswith(".json"):
                continue
            lang = fname[:-5]
            if args.langs and lang not in args.langs:
                continue
            with open(os.path.join(vocab_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            words = data.get("words", [])[:args.sample]
            lang  = data.get("language_code", fname[:-5])
            vocab_words.extend((lang, w) for w in words)
    else:
        print(f"ERROR: vocab directory not found: {vocab_dir}")
        return

    if not vocab_words:
        print("ERROR: no vocabulary words loaded.")
        return

    print(f"Vocab: {len(vocab_words):,} words loaded")
    print(f"Query: {args.image}")

    # Encode query image
    query_emb = encode_image(model, args.image)

    # Encode all vocab words and score
    model.eval()
    scored = []
    for lang, word in vocab_words:
        try:
            t   = word_to_tensor(word, font_path).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = model(t).squeeze(0).cpu().numpy()
            scored.append((cosine_sim(query_emb, emb), lang, word))
        except Exception:
            continue

    scored.sort(reverse=True)
    print(f"\nTop {args.n} nearest vocabulary words:")
    for sim, lang, word in scored[:args.n]:
        print(f"  {sim:+.4f}  [{lang}]  {word}")


def cmd_search_multi(args):
    """
    Search across multiple adapters and/or langpaks simultaneously.
    Results show which file each match came from.
    """
    # Resolve all .pt files to search
    pt_files = []
    if args.adapters:
        for name in args.adapters:
            pt = adapter_path(name)
            if not os.path.exists(pt):
                # Try as direct path
                if os.path.exists(name):
                    pt = name
                else:
                    print(f"WARNING: adapter not found: {name}")
                    continue
            pt_files.append((name, pt))
    if args.langpaks:
        for lp in args.langpaks:
            if not os.path.exists(lp):
                print(f"WARNING: langpak not found: {lp}")
                continue
            name = os.path.splitext(os.path.basename(lp))[0]
            pt_files.append((name, lp))

    if not pt_files:
        print("ERROR: no adapters or langpaks specified.")
        return

    # Load query image using first adapter's model
    first_model = load_model(pt_files[0][1])
    query_emb   = encode_image(first_model, args.image)

    # Search vocab per adapter
    scored = []
    for name, pt in pt_files:
        model = load_model(pt)
        model.eval()

        # Re-encode query with this model
        q_emb = encode_image(model, args.image)

        if args.vocab and os.path.isdir(args.vocab):
            from PIL import ImageFont
            font_path     = _find_search_font()
            cjk_font_path = _find_search_cjk_font()

            for fname in sorted(os.listdir(args.vocab)):
                if not fname.endswith(".json"):
                    continue
                lang = fname[:-5]
                if args.langs and lang not in args.langs:
                    continue
                with open(os.path.join(args.vocab, fname), encoding="utf-8") as f:
                    data = json.load(f)
                words = data.get("words", [])[:args.sample]
                for word in words:
                    try:
                        fp   = cjk_font_path if _is_cjk(word) else font_path
                        size = 28 if _is_cjk(word) else 20
                        font = ImageFont.truetype(fp, size)
                        t    = _render_word(word, font).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            emb = model(t).squeeze(0).cpu().numpy()
                        scored.append((cosine_sim(q_emb, emb), name, lang, word))
                    except Exception:
                        continue

    scored.sort(reverse=True)
    print(f"\nQuery: {args.image}")
    print(f"Top {args.n} results across {len(pt_files)} model(s):")
    for sim, model_name, lang, word in scored[:args.n]:
        print(f"  {sim:+.4f}  [{model_name}]  [{lang}]  {word}")


def _find_search_font():
    candidates = [
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise RuntimeError("No font found.")


def _find_search_cjk_font():
    candidates = [
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return _find_search_font()


def _render_word(word, font):
    from PIL import Image, ImageDraw
    import numpy as np
    img  = Image.new("L", (IMG_W, IMG_H), color=255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), word, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        w, h = IMG_W // 2, IMG_H // 2
    draw.text((max(2, (IMG_W - w) // 2), max(2, (IMG_H - h) // 2)), word, fill=0, font=font)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visual Word Embeddings — script shell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", default=BASE_MODEL, help=f"Base model (default: {BASE_MODEL})")
    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="Add a new script from a folder of images")
    p_add.add_argument("--name",     required=True, help="Adapter name (e.g. coptic)")
    p_add.add_argument("--script",   required=True, help="Script name (e.g. coptic)")
    p_add.add_argument("--images",   required=True, help="Directory of cropped word images")
    p_add.add_argument("--language", default=None,  help="Language name if known")
    p_add.add_argument("--source",   default=None,  help="Source description")
    p_add.add_argument("--epochs",   type=int,   default=20)
    p_add.add_argument("--pairs",    type=int,   default=5000)
    p_add.add_argument("--batch",    type=int,   default=64)
    p_add.add_argument("--lr",       type=float, default=1e-4)
    p_add.add_argument("--unfreeze", type=int,   default=2)
    p_add.add_argument("--vocab",    default="vocabularies",
                        help="Vocabulary dir for replay (default: vocabularies/)")

    # list
    sub.add_parser("list", help="List available adapters")

    # encode
    p_enc = sub.add_parser("encode", help="Encode a single image")
    p_enc.add_argument("--image",   required=True)
    p_enc.add_argument("--adapter", default=None, help="Use adapter instead of base model")

    # neighbours
    p_nb = sub.add_parser("neighbours", help="Find nearest neighbours for a query image")
    p_nb.add_argument("--image",   required=True)
    p_nb.add_argument("--adapter", required=True)
    p_nb.add_argument("--n",       type=int, default=5)

    # search
    p_sr = sub.add_parser("search", help="Find nearest vocabulary words for a scan image")
    p_sr.add_argument("--image",   required=True,              help="Query image")
    p_sr.add_argument("--adapter", default=None,               help="Adapter to use (default: base model)")
    p_sr.add_argument("--vocab",   default="vocabularies",     help="Vocabulary directory (default: vocabularies/)")
    p_sr.add_argument("--sample",  type=int,   default=500,    help="Words per language to search (default: 500)")
    p_sr.add_argument("--langs",   default=None, nargs="+",
                       help="Filter vocab to specific language codes, e.g. --langs zh ja")
    p_sr.add_argument("--n",       type=int,   default=10,     help="Results to show (default: 10)")

    # multi-search
    p_ms = sub.add_parser("multi-search", help="Search across multiple adapters and langpaks")
    p_ms.add_argument("--image",    required=True)
    p_ms.add_argument("--adapters", nargs="+", default=None, help="Adapter names from adapters/")
    p_ms.add_argument("--langpaks", nargs="+", default=None, help="Paths to langpak .pt files")
    p_ms.add_argument("--vocab",    default="vocabularies")
    p_ms.add_argument("--sample",   type=int, default=500)
    p_ms.add_argument("--n",        type=int, default=10)
    p_ms.add_argument("--langs",    nargs="+", default=None)

    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "encode":
        cmd_encode(args)
    elif args.command == "neighbours":
        cmd_neighbours(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "multi-search":
        cmd_search_multi(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
