# visual_word_embeddings — Manual

This document covers all programs in the pipeline. It assumes you know what a terminal is and have a working GPU, or at least a working sense of patience.

---

## visual_embeddings_torch.py

Trains the base model from scratch. This is the file you run once, wait a few hours, and then never touch again unless something went wrong, which it will.

The model takes words, renders them as 384×96 grayscale images, and trains a CNN to produce 256-dimensional vectors. Words that look similar or belong to the same semantic concept across languages end up close together in that vector space. NT-Xent loss, GroupNorm throughout. No BatchNorm — that particular design decision has been learned the hard way.

**Run:**

```bash
python3 visual_embeddings_torch.py
```

Resumes from `visual_embeddings.pt` if it exists. Deletes it first if you want to start over.

**Key constants** (edit in file):

| Constant    | Default | What it does                                      |
|-------------|---------|---------------------------------------------------|
| IMG_W       | 384     | Image width in pixels                             |
| IMG_H       | 96      | Image height in pixels                            |
| FONT_SIZE   | 60      | Base font size for rendering                      |
| EMBED_DIM   | 256     | Embedding vector dimensions                       |
| BATCH_SIZE  | 128     | Training batch size                               |
| EPOCHS      | 100     | Number of training epochs                         |
| LR          | 3e-4    | Learning rate                                     |
| TEMPERATURE | 0.07    | NT-Xent temperature. Lower = sharper, riskier     |

**Pair ratio:** 20% font-variation pairs, 80% semantic concept pairs. This ratio matters. Flip it and the model learns to match visual similarity but forgets that Wasser and 水 are neighbors. The current ratio was arrived at empirically after several hours of GPU time that will not be refunded.

**Outputs:**

`visual_embeddings.pt` — the base model. Contains model weights, embed_dim, img_w, img_h.

**Expected loss curve:**

NT-Xent starts high (around 4.5 at epoch 1) because it is cross-entropy over 2×batch negatives. Below 3.5 by epoch 20, below 3.2 by epoch 100 is reasonable at the current settings. If it does not move at all, something is wrong with the loss function or you are running the old BatchNorm version.

**Validation output** (printed after training):

```
high    sim=+1.000  same word font var
medium  sim=+0.97   en/zh same concept
low     sim=+0.55   unrelated en/de
Ordering: PASS
```

PASS means high > medium > low. That is the minimum bar.

---

## build_vocabulary.py

Downloads Wikipedia word frequency lists and saves them as JSON files in `vocabularies/`. One file per language.

```bash
python3 build_vocabulary.py
```

Outputs `vocabularies/en.json`, `vocabularies/zh.json`, etc. Each file contains a `words` list sorted by frequency. The training pipeline loads all files it finds in that directory.

---

## build_langpak.py

Trains a Langpak — a specialized model for a subset of languages. Used as an embedding anchor for scripts that have no dictionary of their own (see finetune_scan.py, `--reference`).

A Langpak is architecturally identical to the base model but trained only on the languages you specify. It exists so that an ancient script with no known translations can borrow an embedding space from its closest relatives.

**Usage:**

```bash
python build_langpak.py \
    --langs zh ja ko \
    --vocab vocabularies/ \
    --output langpaks/asian.pt

python build_langpak.py \
    --langs ar he \
    --vocab vocabularies/ \
    --output langpaks/semitic.pt
```

**Arguments:**

| Argument   | Required | Default        | Description                              |
|------------|----------|----------------|------------------------------------------|
| --langs    | yes      |                | Language codes (space-separated)         |
| --vocab    | no       | vocabularies/  | Path to vocabulary directory             |
| --output   | yes      |                | Output .pt path                          |
| --epochs   | no       | 50             | Training epochs                          |
| --pairs    | no       | 5000           | Training pairs                           |
| --batch    | no       | 128            | Batch size                               |
| --lr       | no       | 3e-4           | Learning rate                            |
| --embed    | no       | 256            | Embedding dimensions                     |

Semantic concepts are loaded automatically from `visual_embeddings_torch.py` if that file is present in the working directory. If it is not, training still works but only on font-variation pairs, which is suboptimal.

**Output:** A `.pt` file with `tier: langpak` in its metadata. `vwe.py list` will show this.

---

## get_omniglot.py

Downloads Omniglot from GitHub and prepares it for use with `vwe.py add`. Omniglot contains 50 writing systems, all drawn by hand by five different people each. Some of the alphabets were invented by linguists. The model does not know this and does not care.

**List available scripts:**

```bash
python get_omniglot.py
```

**Prepare one script:**

```bash
python get_omniglot.py --alphabet Tifinagh
python get_omniglot.py --alphabet Angelic
```

**Prepare all 30 scripts:**

```bash
python get_omniglot.py --all
```

Images are flattened into `data/<name>/images/` — one image per glyph per person, named `<character>_<source_file>`. No subdirectory structure. Ready for `vwe.py add`.

The zip file is cached locally after the first download. Running it again does not re-download.

---

## scan_dataset.py

Not run directly in normal use. Called by `finetune_scan.py`. Documented here because it defines the data format everything else depends on.

**Manifest format:**

```json
{
    "name":     "coptic_papyrus",
    "script":   "coptic",
    "language": null,
    "source":   "British Museum MS 1234",
    "images":   ["images/0001.png", "images/0002.png"]
}
```

`script` is what determines positive pairs. Two images with the same script are pushed together. `language` and `source` are informational only. `language` can be null for ancient or undeciphered scripts.

**Image requirements:** Any common format (PNG, JPG, TIFF, BMP). Grayscale or color — converted to grayscale on load. Any size — resized to 384×96 with LANCZOS. Pre-cropped to one word or glyph per file.

**Tensor output:** (1, 96, 384) float32, 0=black 1=white. Identical to what `visual_embeddings_torch.py` produces. They can be mixed in the same DataLoader.

**Smoke test:**

```bash
python scan_dataset.py data/tifinagh/manifest.json
```

Prints shape, range, and whether both pass.

---

## finetune_scan.py

Fine-tunes a base model or langpak on scan data. The base model's early CNN layers (which detect edges and strokes, universal across all scripts) are frozen. Only the later layers and projection head are trained on the new script.

The result is a new `.pt` file. The base model is not modified.

**Usage:**

```bash
python finetune_scan.py \
    --base visual_embeddings.pt \
    --manifests data/tifinagh/manifest.json \
    --output adapters/tifinagh.pt

# With vocab replay (prevents forgetting of base languages)
python finetune_scan.py \
    --base visual_embeddings.pt \
    --manifests data/tifinagh/manifest.json \
    --output adapters/tifinagh.pt \
    --vocab vocabularies/

# Alt C: bootstrap against a langpak instead of the base model
python finetune_scan.py \
    --base visual_embeddings.pt \
    --reference langpaks/asian.pt \
    --manifests data/mystery_script/manifest.json \
    --output adapters/mystery.pt

# Multiple manifests (train on several scripts at once)
python finetune_scan.py \
    --base visual_embeddings.pt \
    --manifests data/coptic/manifest.json data/tifinagh/manifest.json \
    --output adapters/northafrica.pt
```

**Arguments:**

| Argument    | Required | Default | Description                                            |
|-------------|----------|---------|--------------------------------------------------------|
| --base      | yes      |         | Base model .pt                                         |
| --manifests | yes      |         | One or more manifest.json paths                        |
| --output    | yes      |         | Output adapter .pt                                     |
| --reference | no       | none    | Langpak .pt to use as anchor (Alt C, no dict required) |
| --epochs    | no       | 20      | Training epochs                                        |
| --pairs     | no       | 5000    | Training pairs                                         |
| --batch     | no       | 64      | Batch size                                             |
| --lr        | no       | 1e-4    | Learning rate                                          |
| --unfreeze  | no       | 2       | Freeze first N conv blocks (0=train all, 4=train only projection) |
| --vocab     | no       | none    | Vocabulary dir for replay during fine-tuning           |

**On `--unfreeze`:** The CNN has 4 conv blocks. `--unfreeze 2` freezes the first two (edge/stroke detectors) and trains the last two. `--unfreeze 3` is more conservative and recommended if the new script is very different from the training languages and the dataset is small. `--unfreeze 0` trains everything, which is only appropriate with large datasets and a good reason.

**On `--reference`:** When a script has no dictionary, there are no labels to build cross-lingual pairs from. `--reference` solves this by bootstrapping against a langpak. The langpak's embedding space provides the geometry; the adapter learns to place the new script within it. The base model is still used for the architecture — `--reference` only affects which weights are loaded, not the `--base` argument, which remains required.

---

## vwe.py

The main interface. Wraps everything else into a single command-line tool.

```bash
python vwe.py <command> [arguments]
```

---

### vwe.py add

Prepares a manifest, runs `finetune_scan.py`, and saves the result to `adapters/`.

```bash
python vwe.py add \
    --name tifinagh \
    --script tifinagh \
    --images data/tifinagh/images/
```

**Arguments:**

| Argument    | Required | Default        | Description                          |
|-------------|----------|----------------|--------------------------------------|
| --name      | yes      |                | Adapter name (used as filename)      |
| --script    | yes      |                | Script identifier                    |
| --images    | yes      |                | Directory of cropped word images     |
| --language  | no       | null           | Language name if known               |
| --source    | no       | null           | Source description                   |
| --base      | no       | visual_embeddings.pt | Base model path               |
| --epochs    | no       | 20             | Training epochs                      |
| --pairs     | no       | 5000           | Training pairs                       |
| --batch     | no       | 64             | Batch size                           |
| --lr        | no       | 1e-4           | Learning rate                        |
| --unfreeze  | no       | 2              | Conv blocks to freeze                |
| --vocab     | no       | vocabularies/  | Vocabulary dir for replay            |

---

### vwe.py list

Lists all adapters in `adapters/`. Shows name, tier (adapter or langpak), script, and number of images.

```bash
python vwe.py list
```

---

### vwe.py encode

Encodes a single image and prints the embedding vector. Useful for debugging and for verifying that a model loaded correctly.

```bash
python vwe.py encode --image data/tifinagh/images/0001.png
python vwe.py encode --image data/tifinagh/images/0001.png --adapter tifinagh
```

---

### vwe.py neighbours

Finds the nearest images within the same adapter's manifest. Searches among scanned images only, not vocabulary words.

```bash
python vwe.py neighbours \
    --image data/tifinagh/images/character04_0913_03.png \
    --adapter tifinagh \
    --n 10
```

---

### vwe.py search

Finds the nearest vocabulary words for a scanned image. Renders words from the vocabulary files and scores them against the query image using the adapter model.

```bash
python vwe.py search \
    --image data/tifinagh/images/character04_0913_03.png \
    --adapter tifinagh \
    --vocab vocabularies/

# Filter to specific languages
python vwe.py search \
    --image data/tifinagh/images/character04_0913_03.png \
    --adapter tifinagh \
    --vocab vocabularies/ \
    --langs zh ja \
    --sample 5000 \
    --n 10
```

**Arguments:**

| Argument  | Required | Default       | Description                              |
|-----------|----------|---------------|------------------------------------------|
| --image   | yes      |               | Query image path                         |
| --adapter | no       | base model    | Adapter name or none for base model      |
| --vocab   | no       | vocabularies/ | Vocabulary directory                     |
| --sample  | no       | 500           | Words per language to score              |
| --n       | no       | 10            | Number of results to show                |
| --langs   | no       | all           | Filter to specific language codes        |

Higher `--sample` finds better matches but takes longer. 500 is fast. 5000 is thorough. 50000 is an act of faith.

---

### vwe.py multi-search

Searches across multiple adapters and langpaks simultaneously. Results are ranked globally and show which model found each match.

```bash
python vwe.py multi-search \
    --image data/tifinagh/images/character04_0913_03.png \
    --adapters tifinagh \
    --langpaks langpaks/asian.pt \
    --vocab vocabularies/ \
    --langs zh ja \
    --sample 1000 \
    --n 15
```

**Arguments:**

| Argument   | Required | Default       | Description                              |
|------------|----------|---------------|------------------------------------------|
| --image    | yes      |               | Query image path                         |
| --adapters | no       |               | Adapter names from adapters/ (space-sep) |
| --langpaks | no       |               | Paths to langpak .pt files               |
| --vocab    | no       | vocabularies/ | Vocabulary directory                     |
| --sample   | no       | 500           | Words per language per model             |
| --n        | no       | 10            | Results to show                          |
| --langs    | no       | all           | Filter to specific language codes        |

At least one of `--adapters` or `--langpaks` must be specified. Combining both is the intended use case.

---

## Typical workflows

**Start from scratch:**

```bash
python3 build_vocabulary.py
python3 visual_embeddings_torch.py
```

**Add a script from Omniglot:**

```bash
python get_omniglot.py --alphabet Tifinagh
python vwe.py add --name tifinagh --script tifinagh --images data/tifinagh/images/
python vwe.py search --image data/tifinagh/images/character04_0913_03.png --adapter tifinagh --vocab vocabularies/
```

**Add a script using a langpak as reference:**

```bash
python build_langpak.py --langs zh ja ko --vocab vocabularies/ --output langpaks/asian.pt
python finetune_scan.py \
    --base visual_embeddings.pt \
    --reference langpaks/asian.pt \
    --manifests data/mystery_script/manifest.json \
    --output adapters/mystery.pt
python vwe.py multi-search \
    --image data/mystery_script/images/0001.png \
    --adapters mystery \
    --langpaks langpaks/asian.pt \
    --vocab vocabularies/ --langs zh ja
```

**List everything:**

```bash
python vwe.py list
```

---

## File structure

```
visual_word_embeddings/
  visual_embeddings_torch.py   Base model training
  build_vocabulary.py          Wikipedia vocabulary builder
  build_langpak.py             Langpak trainer
  get_omniglot.py              Omniglot downloader
  scan_dataset.py              Scan data loader
  finetune_scan.py             Fine-tuning on scan data
  vwe.py                       Main interface

  visual_embeddings.pt         Base model (generated)
  vocabularies/                Vocabulary JSON files (generated)
    en.json
    zh.json
    ...
  adapters/                    Adapter .pt files (generated)
    tifinagh.pt
    tifinagh_manifest.json
    ...
  langpaks/                    Langpak .pt files (generated)
    asian.pt
    ...
  data/                        Scan data (you provide)
    tifinagh/
      images/
        0001.png
        ...
```
