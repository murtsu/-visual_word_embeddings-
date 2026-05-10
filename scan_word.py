"""
scan_word.py — Visual word scanner
Renders one query word → embedding vector → finds closest visual match
per loaded language vocabulary. No semantics. No dictionaries. Pure vectors.

Run:
    python3 scan_word.py --word water --lang en
    python3 scan_word.py --word 水 --lang zh --top 5000
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# ── Re-use the encoder and rendering from visual_embeddings_torch.py ─────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visual_embeddings_torch import (
    VisualWordEncoder, word_to_tensor, DEVICE, EMBED_DIM,
    IMG_W, IMG_H, FONT_PATH, CJK_FONT_PATH,
    MULTILINGUAL_WORDS
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "visual_embeddings.pt"
VOCAB_DIR   = "vocabularies"
TOP_WORDS   = 3000     # words to scan per language (speed vs. coverage)
BATCH_SIZE  = 100     # forward-pass batch size

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Train first with:  python3 visual_embeddings_torch.py"
        )
    ckpt = torch.load(path, map_location=DEVICE)
    model = VisualWordEncoder(embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded: {path}  (embed_dim={ckpt['embed_dim']})")
    return model

# ── BATCH ENCODE ──────────────────────────────────────────────────────────────

def batch_encode(model, words, font_size=20, desc="encoding"):
    """Encode a list of word strings → (N, embed_dim) numpy array."""
    embeddings = []
    for i in tqdm(range(0, len(words), BATCH_SIZE), desc=f"  {desc}", leave=False):
        batch_words = words[i : i + BATCH_SIZE]
        tensors = []
        for w in batch_words:
            try:
                t = word_to_tensor(w, font_size)
            except Exception:
                t = torch.zeros(1, IMG_H, IMG_W)
            tensors.append(t)
        batch = torch.stack(tensors).to(DEVICE)   # (B, 1, H, W)
        with torch.no_grad():
            emb = model(batch)                     # (B, embed_dim)
        embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

# ── LOAD VOCABS ───────────────────────────────────────────────────────────────

def load_vocabularies(vocab_dir=VOCAB_DIR, top=TOP_WORDS):
    result = {}
    if os.path.exists(vocab_dir):
        for fname in sorted(os.listdir(vocab_dir)):
            if fname.endswith(".json"):
                lang = fname[:-5]
                with open(os.path.join(vocab_dir, fname), encoding="utf-8") as f:
                    data = json.load(f)
                result[lang] = data["words"][:top]
        if result:
            print(f"  Wikipedia vocabs: {list(result.keys())}")
            return result
    # Fallback to built-in
    print("  No vocabularies dir — using built-in word lists")
    for lang_name, words in MULTILINGUAL_WORDS.items():
        result[lang_name] = words
    return result

# ── WORD RENDER → PIL IMAGE ───────────────────────────────────────────────────

def render_word_image(word, font_size=28, width=IMG_W*3, height=IMG_H*2):
    """Larger image for display."""
    from visual_embeddings_torch import is_cjk, reshape_arabic
    word_r = reshape_arabic(word)
    path = CJK_FONT_PATH if is_cjk(word_r) else FONT_PATH
    try:
        font = ImageFont.truetype(path, font_size)
    except Exception:
        font = ImageFont.load_default()
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), word_r, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        w, h = width // 2, font_size
    x = max(2, (width - w) // 2)
    y = max(2, (height - h) // 2)
    draw.text((x, y), word_r, fill=0, font=font)
    return img

# ── SCAN ──────────────────────────────────────────────────────────────────────

def scan(model, query_word, query_lang, vocabs, top_n=5):
    """
    Returns dict: lang → [(sim, word), ...]  top_n matches per language,
    excluding the query language itself.
    """
    print(f"\n  Query: '{query_word}' ({query_lang})")

    # Encode query
    qt = word_to_tensor(query_word).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_emb = model(qt).cpu().numpy()[0]   # (embed_dim,)

    results = {}
    for lang, words in vocabs.items():
        if not words:
            continue
        embs = batch_encode(model, words, desc=lang)
        # cosine similarity: embeddings are L2-normalised → dot product = cosine sim
        sims = embs @ q_emb                  # (N,)
        top_idx = np.argsort(sims)[::-1][:top_n]
        results[lang] = [(float(sims[i]), words[i]) for i in top_idx]

    return q_emb, results

# ── PLOT ──────────────────────────────────────────────────────────────────────

def plot_results(query_word, query_lang, q_emb, results, out_path="scan_result.png"):
    langs = list(results.keys())
    n_langs = len(langs)

    fig = plt.figure(figsize=(14, 3 + n_langs * 1.4), facecolor="#0d0d0d")
    fig.suptitle(
        f'Visual nearest neighbour scan   \u00b7   query: \u201c{query_word}\u201d ({query_lang})',
        color="#e8e8e8", fontsize=13, y=0.98, fontfamily="monospace"
    )

    # Grid: query column | lang results
    gs = gridspec.GridSpec(
        n_langs + 1, 4,
        figure=fig,
        hspace=0.08, wspace=0.04,
        top=0.93, bottom=0.03, left=0.02, right=0.98
    )

    # ── Query word ────────────────────────────────────────────────────────────
    ax_q = fig.add_subplot(gs[0, 0])
    q_img = render_word_image(query_word)
    ax_q.imshow(q_img, cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax_q.set_title(f"query · {query_lang}", color="#aaaaaa",
                   fontsize=8, fontfamily="monospace", pad=3)
    ax_q.axis("off")

    # Similarity bar reference (full width = 1.0)
    ax_ref = fig.add_subplot(gs[0, 1:4])
    ax_ref.set_xlim(0, 1)
    ax_ref.set_ylim(0, 1)
    ax_ref.axis("off")
    ax_ref.text(0.5, 0.5, "cosine similarity  →  1.0",
                ha="center", va="center", color="#555555",
                fontsize=7, fontfamily="monospace")

    # ── Per-language rows ─────────────────────────────────────────────────────
    ACCENT = "#00e5cc"
    DIM    = "#1e1e1e"

    for row, lang in enumerate(langs, start=1):
        matches = results[lang]
        top_sim, top_word = matches[0]

        # Word image
        ax_img = fig.add_subplot(gs[row, 0])
        img = render_word_image(top_word)
        ax_img.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax_img.set_ylabel(lang, color="#777777", fontsize=7,
                          fontfamily="monospace", rotation=0,
                          labelpad=28, va="center")
        ax_img.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
        for spine in ax_img.spines.values():
            spine.set_edgecolor("#333333")

        # Similarity bar + word label
        ax_bar = fig.add_subplot(gs[row, 1:4])
        ax_bar.set_facecolor(DIM)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(0, 1)
        ax_bar.axis("off")

        bar_w = max(0.01, min(1.0, top_sim))
        # Background track
        ax_bar.barh(0.5, 1.0, height=0.55, left=0,
                    color="#1a1a1a", align="center")
        # Filled bar
        color = ACCENT if top_sim > 0.6 else ("#f0a500" if top_sim > 0.35 else "#e04040")
        ax_bar.barh(0.5, bar_w, height=0.55, left=0,
                    color=color, alpha=0.85, align="center")

        # Labels
        ax_bar.text(bar_w + 0.01, 0.5, f"{top_sim:.3f}",
                    va="center", ha="left", color=color,
                    fontsize=8, fontfamily="monospace")
        ax_bar.text(0.97, 0.5, top_word,
                    va="center", ha="right", color="#cccccc",
                    fontsize=9, fontfamily="monospace")

        # Runners-up (dim)
        runners = "  ".join(f"{w}({s:.2f})" for s, w in matches[1:])
        ax_bar.text(0.01, 0.05, runners,
                    va="bottom", ha="left", color="#444444",
                    fontsize=6, fontfamily="monospace")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved: {out_path}")
    return out_path

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visual word scanner")
    parser.add_argument("--word",  default="water",  help="Query word")
    parser.add_argument("--lang",  default="en",     help="Query language code")
    parser.add_argument("--top",   type=int, default=TOP_WORDS,
                        help="Words to scan per language")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--out",   default="scan_result.png")
    args = parser.parse_args()

    print("=" * 55)
    print("VISUAL WORD SCANNER")
    print(f"  query : '{args.word}' ({args.lang})")
    print(f"  top-N : {args.top} words per language")
    print(f"  device: {DEVICE}")
    print("=" * 55)

    model  = load_model(args.model)
    vocabs = load_vocabularies(top=args.top)

    q_emb, results = scan(model, args.word, args.lang, vocabs, top_n=4)

    print("\n  Results:")
    for lang, matches in results.items():
        top = matches[0]
        print(f"    {lang:6}  {top[1]:20}  sim={top[0]:.3f}")

    out = plot_results(args.word, args.lang, q_emb, results, out_path=args.out)

if __name__ == "__main__":
    main()
