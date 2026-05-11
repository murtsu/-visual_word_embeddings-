"""
Fine-tune visual_word_embeddings on scan data.

Loads a trained base model, freezes early CNN layers,
trains only the later layers on new script data.
Saves a new .pt file — base model untouched.

Usage:
    python finetune_scan.py \
        --base visual_embeddings.pt \
        --manifests data/coptic/manifest.json \
        --output coptic_finetuned.pt
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont

from scan_dataset import ScanPairDataset, IMG_W, IMG_H

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MARGIN = 1.0


# ─── MODEL ───────────────────────────────────────────────────────────────────
# GroupNorm to match the trained base model.

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


# ─── LOSS ────────────────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, labels):
        dist = F.pairwise_distance(emb_a, emb_b)
        pos_loss = labels * dist.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return (pos_loss + neg_loss).mean()


# ─── VOCAB REPLAY ────────────────────────────────────────────────────────────
# Mix in rendered base vocabulary pairs during fine-tuning so CJK and other
# scripts don't drift out of the embedding space.

def _find_font():
    candidates = [
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise RuntimeError("No font found.")

def _find_cjk_font():
    candidates = [
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return _find_font()

def _word_to_tensor(word, font_path, font_size=20):
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

def _is_cjk(word):
    for ch in word:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7A3:
            return True
    return False


class VocabReplayDataset(Dataset):
    """
    Renders word pairs from vocabulary JSON files to keep the base
    embedding space stable during fine-tuning.
    """

    def __init__(self, vocab_dir, n_pairs=2000, words_per_lang=300):
        self.pairs = []

        font      = _find_font()
        cjk_font  = _find_cjk_font()

        # Load words
        by_lang = {}
        for fname in sorted(os.listdir(vocab_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(vocab_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            lang  = data.get("language_code", fname[:-5])
            words = data.get("words", [])[:words_per_lang]
            if words:
                by_lang[lang] = words

        if not by_lang:
            print("  WARNING: no vocab files found, skipping replay.")
            return

        all_words = [(lang, w) for lang, ws in by_lang.items() for w in ws]
        langs     = list(by_lang.keys())
        n_pos     = n_pairs // 2
        n_neg     = n_pairs - n_pos

        print(f"  VocabReplay: {len(all_words):,} words across {len(langs)} languages")

        # Positive: same word, font size variation
        rendered = {}
        def get_tensor(word):
            if word not in rendered:
                fp        = cjk_font if _is_cjk(word) else font
                font_size = 28 if _is_cjk(word) else 20
                try:
                    rendered[word] = _word_to_tensor(word, fp, font_size)
                except Exception:
                    rendered[word] = None
            return rendered[word]

        for _ in range(n_pos):
            lang, word = random.choice(all_words)
            ta = get_tensor(word)
            tb = get_tensor(word)
            if ta is None or tb is None:
                continue
            self.pairs.append((ta, tb, torch.tensor(1.0)))

        # Negative: words from different languages
        for _ in range(n_neg):
            if len(langs) < 2:
                break
            l1, l2 = random.sample(langs, 2)
            w1 = random.choice(by_lang[l1])
            w2 = random.choice(by_lang[l2])
            ta = get_tensor(w1)
            tb = get_tensor(w2)
            if ta is None or tb is None:
                continue
            self.pairs.append((ta, tb, torch.tensor(0.0)))

        random.shuffle(self.pairs)
        print(f"  VocabReplay: {len(self.pairs):,} pairs ready")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ─── FREEZE ──────────────────────────────────────────────────────────────────

def freeze_base_layers(model, n_freeze):
    modules = list(model.cnn.children())
    cutoff = n_freeze * 4  # 4 modules per conv block
    for i, module in enumerate(modules):
        for param in module.parameters():
            param.requires_grad = (i >= cutoff)
    frozen    = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Frozen params:    {frozen:,}")
    print(f"  Trainable params: {trainable:,}")


# ─── LOAD / SAVE ─────────────────────────────────────────────────────────────

def load_base(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = VisualWordEncoder(embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"Base model loaded: {path}  (embed_dim={ckpt['embed_dim']})")
    return model, ckpt


def save_finetuned(model, base_ckpt, output_path, manifests, n_freeze):
    torch.save({
        "model_state":   model.state_dict(),
        "embed_dim":     base_ckpt["embed_dim"],
        "img_w":         base_ckpt["img_w"],
        "img_h":         base_ckpt["img_h"],
        "finetuned_on":  manifests,
        "frozen_blocks": n_freeze,
    }, output_path)
    print(f"Saved: {output_path}")


# ─── TRAIN ───────────────────────────────────────────────────────────────────

def finetune(model, dataset, epochs, batch_size, lr):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for ta, tb, labels in loader:
            ta, tb, labels = ta.to(DEVICE), tb.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(ta), model(tb), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return model


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune on scan data")
    parser.add_argument("--base",      required=True)
    parser.add_argument("--manifests", required=True, nargs="+")
    parser.add_argument("--output",    required=True)
    parser.add_argument("--epochs",    type=int,   default=20)
    parser.add_argument("--pairs",     type=int,   default=5000)
    parser.add_argument("--batch",     type=int,   default=64)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--unfreeze",  type=int,   default=2)
    parser.add_argument("--vocab",     default=None,
                        help="Vocabulary dir for replay (default: none). "
                             "Pass vocabularies/ to prevent CJK forgetting.")
    args = parser.parse_args()

    print("=" * 60)
    print("FINE-TUNE ON SCAN DATA")
    print("=" * 60)
    print(f"Base:      {args.base}")
    print(f"Manifests: {args.manifests}")
    print(f"Output:    {args.output}")
    print(f"Epochs:    {args.epochs}  Pairs: {args.pairs}  Batch: {args.batch}")
    print(f"LR:        {args.lr}  Freeze blocks: {args.unfreeze}")
    print(f"Device:    {DEVICE}")
    print()

    model, base_ckpt = load_base(args.base)

    print("\nFreezing layers...")
    freeze_base_layers(model, args.unfreeze)

    print("\nBuilding dataset...")
    dataset = ScanPairDataset(args.manifests, n_pairs=args.pairs)

    if args.vocab and os.path.isdir(args.vocab):
        print("Building vocab replay dataset...")
        replay  = VocabReplayDataset(args.vocab, n_pairs=args.pairs // 2)
        dataset = ConcatDataset([dataset, replay])
        print(f"  Total pairs: {len(dataset):,}")

    print(f"\n=== FINE-TUNING on {DEVICE} ===")
    model = finetune(model, dataset, args.epochs, args.batch, args.lr)
    model.eval()

    save_finetuned(model, base_ckpt, args.output, args.manifests, args.unfreeze)
    print("\nDone.")


if __name__ == "__main__":
    main()
