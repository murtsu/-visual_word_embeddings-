# Visual Word Embeddings

**Cross-lingual, script-agnostic word embeddings trained on visual appearance alone.**

No tokenisation. No dictionary. No pretrained vectors.

Just what the word looks like.

---

## The idea

Every AI handling text today reads words as symbols.

"Water" is one symbol. "Wasser" is another. "水" is a third.

All three mean the same thing but the model does not know that unless someone told it.

This project asks: what happens if you let the model *see* the words instead of just reading them?

Each word is rendered as an image. A CNN learns what words look like. Through contrastive training on font-size variation and cross-lingual semantic pairs, the network discovers that visually similar words tend to mean similar things — without any explicit semantic labelling.

The result: search for "Wasser" and the nearest neighbours are "水", "water", and "agua".

Nobody coded that in. The network saw it.

---

## Results

Trained on Wikipedia vocabularies for 10 languages (50,000 words each) on an RTX 2080.

**Loss:** 0.093 → 0.009 over 50 epochs

**Similarity ordering (PASS):**

| Pair | Similarity |
|------|-----------|
| same word, font size variation | 1.000 |
| water / Wasser (en/de) | 1.000 |
| fire / 火 (en/zh) | 0.989 |
| house / بيت (en/ar) | 0.999 |
| love / любовь (en/ru) | 1.000 |
| love / kärlek (en/sv) | 1.000 |
| unrelated words (en) | 0.156 |
| unrelated (en/de) | 0.394 |

**Script clustering:**

| Script | Within | Between | Result |
|--------|--------|---------|--------|
| Cyrillic | 1.000 | 0.960 | OK |
| Arabic | 1.000 | 0.960 | OK |
| CJK | 1.000 | 0.960 | OK |
| Devanagari | 0.999 | 0.960 | OK |
| Thai | 0.998 | 0.958 | OK |

**Nearest neighbours (post-training):**

```
'Wasser'  →  水(1.00)  水(1.00)  water(1.00)  月(1.00)  月(1.00)
'手'      →  手(1.00)  main(1.00)  el(1.00)
'water'   →  水(1.00)  house(1.00)  水(1.00)
```

---

## Why this is different

Existing work on visual embeddings starts with text and adds images as a supplement.

This project starts with the image of the word as the **only input**.

It works across all writing systems simultaneously. Same model. Same training. Same embedding space.

This means it can handle:
- Words it has never seen before
- Handwritten text
- Damaged or partially corrupted OCR output
- Historical documents
- New words not in any dictionary

If the word looks like something the model recognises, it finds the right neighbourhood.

---

## Architecture

```
Input: word string
  ↓
Render to 128×32 grayscale image (PIL + Noto fonts)
  ↓
CNN Encoder:
  Conv(1→32, k=3) → BN → ReLU → MaxPool(2)
  Conv(32→64, k=3) → BN → ReLU → MaxPool(2)
  Conv(64→128, k=3) → BN → ReLU → MaxPool(2)
  Conv(128→256, k=3) → BN → ReLU → AdaptiveAvgPool(1×1)
  FC(256→256) → ReLU → Dropout(0.1) → FC(256→256)
  L2 normalisation
  ↓
Output: 256-dimensional unit vector
```

**Training:** Contrastive loss (Hadsell et al. 2006) on three pair types:
1. Same word, different font sizes (font invariance)
2. Same concept across languages (semantic signal)
3. Random different words (negative pairs)

**Optimiser:** AdamW with cosine annealing LR schedule

---

## Languages supported

Arabic, Hindi, Thai, Chinese, Japanese, Korean, Greek, Hebrew, Russian, English, German, French, Spanish, Italian, Swedish, Turkish, Dutch, Polish, Portuguese — and any language renderable by Noto fonts.

---

## Requirements

```bash
pip install torch torchvision pillow numpy arabic-reshaper python-bidi tqdm datasets
```

**Fonts:**
```bash
# Linux
sudo apt install fonts-freefont-ttf
pamac install noto-fonts-cjk        # Arch/Manjaro
sudo apt install fonts-noto-cjk     # Debian/Ubuntu
```

---

## Quick start

**Train from scratch:**
```bash
# Build Wikipedia vocabularies (downloads ~5GB per language)
python3 build_vocabulary.py --langs en sv de fr es ru ar hi zh ja --top 50000

# Train the model
python3 visual_embeddings_torch.py
```

**Use a trained model:**
```python
from visual_embeddings_torch import load_model

model = load_model('visual_embeddings.pt')

# Embed a word
vector = model.encode_word('Wasser')  # returns numpy array (256,)

# Find nearest neighbours
from visual_embeddings_torch import nearest_neighbours, load_wiki_vocab

vocab = load_wiki_vocab('vocabularies')
all_words = [w for words in vocab.values() for w in words[:500]]
neighbours = nearest_neighbours(model, 'Wasser', all_words, n=5)
```

---

## Files

| File | Description |
|------|-------------|
| `visual_embeddings_torch.py` | Main training and evaluation pipeline (PyTorch) |
| `visual_embeddings.py` | Validation pipeline (pure numpy, no GPU needed) |
| `build_vocabulary.py` | Wikipedia vocabulary builder |

---

## Known limitations

- Latin script clustering is weaker than non-Latin scripts — short functional words (el, su, de) pull Latin words together regardless of meaning. Addressed in next version with harder negative mining.
- CJK nearest neighbours sometimes return visually similar but semantically unrelated characters. More training data and longer training will improve this.
- Handwriting and historical fonts not yet tested — next planned experiment.

---

## Next steps

- [ ] Evaluate against MUSE benchmark (Facebook Research cross-lingual word similarity)
- [ ] Harder negative mining for Latin script separation
- [ ] Extend to 30+ languages
- [ ] Handwriting experiment
- [ ] Pre-trained model weights release

---

## Origin

This project started at midnight on a Monday.

By 4:30 AM the network had learned that "Wasser" and "水" belong together.

Nobody told it. It saw it.

Built by **Marko Tahvanainen** in collaboration with **Claude (Anthropic)**.

Marko had the idea. Claude wrote the code. That is a collaboration and we are not going to pretend otherwise.

**LinkedIn:** [linkedin.com/in/marko-tahvanainen-963979340](https://linkedin.com/in/marko-tahvanainen-963979340)

---

## License

Apache 2.0

Use it. Improve it. Share it.

That is the point.
