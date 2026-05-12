"""
build_langpak.py — Train a Langpak from vocabulary files.

A Langpak is a .pt file trained on a specific set of languages.
It serves as a reference embedding space for adapters that lack
their own dictionary (Alt C bootstrapping).

Usage:
    python build_langpak.py \
        --langs zh ja ko \
        --vocab vocabularies/ \
        --output langpaks/asian.pt

    python build_langpak.py \
        --langs ar he \
        --vocab vocabularies/ \
        --output langpaks/semitic.pt
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPERATURE = 0.07
IMG_W       = 384
IMG_H       = 96
FONT_SIZE   = 60


# ─── MODEL (must match base) ──────────────────────────────────────────────────

class VisualWordEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.projection(self.cnn(x)), p=2, dim=1)


# ─── LOSS ─────────────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a, emb_b):
        batch_size = emb_a.size(0)
        emb = torch.cat([emb_a, emb_b], dim=0)
        sim = torch.mm(emb, emb.t()) / self.temperature
        mask = torch.eye(2 * batch_size, device=emb.device).bool()
        sim.masked_fill_(mask, float("-inf"))
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=emb.device),
            torch.arange(batch_size, device=emb.device),
        ])
        return F.cross_entropy(sim, labels)


# ─── FONT ─────────────────────────────────────────────────────────────────────

def find_font(cjk=False):
    cjk_candidates = [
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    latin_candidates = [
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    candidates = cjk_candidates + latin_candidates if cjk else latin_candidates + cjk_candidates
    for p in candidates:
        if os.path.exists(p):
            return p
    raise RuntimeError("No font found.")

def is_cjk(word):
    for ch in word:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7A3:
            return True
    return False

def word_to_tensor(word, font_path, font_size=FONT_SIZE):
    font = ImageFont.truetype(font_path, font_size)
    img  = Image.new("L", (IMG_W, IMG_H), color=255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), word, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        w, h = int(font.getlength(word)), font_size
    draw.text((max(2, (IMG_W - w) // 2), max(2, (IMG_H - h) // 2)), word, fill=0, font=font)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ─── DATASET ──────────────────────────────────────────────────────────────────

class LangpakDataset(Dataset):
    """
    Positive pairs from selected languages only.
    Same word rendered twice (font size variation) + cross-lang concept pairs.
    """
    def __init__(self, langs, vocab_dir, concepts, n_pairs=5000):
        self.pairs = []
        font      = find_font(cjk=False)
        cjk_font  = find_font(cjk=True)
        font_sizes = [50, 54, 58, 62, 66, 70, 74]

        # Load vocab for selected langs only
        by_lang = {}
        for fname in sorted(os.listdir(vocab_dir)):
            if not fname.endswith(".json"):
                continue
            lang = fname[:-5]
            if lang not in langs:
                continue
            with open(os.path.join(vocab_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            by_lang[lang] = data["words"]
            print(f"  Loaded {lang}: {len(data['words']):,} words")

        if not by_lang:
            raise ValueError(f"No vocab files found for langs: {langs}")

        word_list = [w for ws in by_lang.values() for w in ws[:5000]]

        # Filter concepts to pairs where both words are in selected langs
        lang_set = set(langs)
        filtered_concepts = []
        for name, translations in concepts:
            filtered = translations  # use all, cross-lang signal from mixing
            if len(filtered) >= 2:
                filtered_concepts.append(filtered)

        raw_pairs = []
        for _ in range(n_pairs // 5):
            word = random.choice(word_list)
            sa, sb = random.choice(font_sizes), random.choice(font_sizes)
            raw_pairs.append((word, sa, word, sb))
        for _ in range(n_pairs * 4 // 5):
            if filtered_concepts:
                translations = random.choice(filtered_concepts)
                a = random.choice(translations)
                b = random.choice(translations)
                raw_pairs.append((a, FONT_SIZE, b, FONT_SIZE))

        random.shuffle(raw_pairs)

        print(f"  Pre-rendering {len(raw_pairs):,} pairs...")
        rendered = {}
        def get_t(word, size):
            key = (word, size)
            if key not in rendered:
                fp = cjk_font if is_cjk(word) else font
                try:
                    rendered[key] = word_to_tensor(word, fp, size)
                except Exception:
                    rendered[key] = None
            return rendered[key]

        for wa, sa, wb, sb in tqdm(raw_pairs, desc="  rendering"):
            ta = get_t(wa, sa)
            tb = get_t(wb, sb)
            if ta is not None and tb is not None:
                self.pairs.append((ta, tb))
        print(f"  {len(self.pairs):,} pairs ready")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ─── TRAIN ────────────────────────────────────────────────────────────────────

def train(model, dataset, epochs, batch_size, lr):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)
    criterion = NTXentLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for ta, tb in loader:
            ta, tb = ta.to(DEVICE), tb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(ta), model(tb))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        avg = total / len(loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
    return model


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a Langpak")
    parser.add_argument("--langs",   required=True, nargs="+", help="Language codes to include")
    parser.add_argument("--vocab",   default="vocabularies",   help="Vocabulary directory")
    parser.add_argument("--output",  required=True,            help="Output .pt file")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--pairs",   type=int,   default=5000)
    parser.add_argument("--batch",   type=int,   default=128)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--embed",   type=int,   default=256)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print("=" * 60)
    print("BUILD LANGPAK")
    print("=" * 60)
    print(f"Languages: {args.langs}")
    print(f"Output:    {args.output}")
    print(f"Epochs:    {args.epochs}  Pairs: {args.pairs}  Batch: {args.batch}")
    print(f"Device:    {DEVICE}")
    print()

    # Import concepts from training file if available
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("vwe_train", "visual_embeddings_torch.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        concepts = mod.SEMANTIC_CONCEPTS
        print(f"  Concepts: {len(concepts)} loaded from visual_embeddings_torch.py")
    except Exception:
        concepts = []
        print("  Concepts: none (visual_embeddings_torch.py not found)")

    model = VisualWordEncoder(embed_dim=args.embed).to(DEVICE)
    dataset = LangpakDataset(args.langs, args.vocab, concepts, n_pairs=args.pairs)
    model = train(model, dataset, args.epochs, args.batch, args.lr)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "embed_dim":   args.embed,
        "img_w":       IMG_W,
        "img_h":       IMG_H,
        "langs":       args.langs,
        "tier":        "langpak",
    }, args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
