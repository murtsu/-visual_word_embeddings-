# visual_word_embeddings

Cross-lingual, script-agnostic word embeddings trained on visual appearance alone.

No tokenizer. No dictionary. No pretrained vectors. No labels.

Type `water` and the model finds `agua`, `水`, and `su`. It does not know those words mean the same thing. It figured it out by looking at the shapes.

---

## How it works

Every word gets rendered to a small grayscale image. A CNN reads the image and produces a 256-dimensional vector. Words that look visually similar, or that the model has learned to associate through cross-lingual training pairs, end up close together in that vector space.

The model never sees text as text. It sees pixels.

---

## Results

After training on 10 languages with 100,000 words each:

```
Query: water (en)
  agua  (es)  sim=1.00
  水    (zh)  sim=0.99
  su    (tr)  sim=0.99
  水    (ja)  sim=0.99

Query: Wasser (de)
  water (en)  sim=0.99
  水    (zh)  sim=0.98
  agua  (es)  sim=0.98

Query: 手 (zh)
  手    (ja)  sim=1.00
  main  (fr)  sim=0.99
  el    (es)  sim=0.99

Query: dag (sv)
  日    (ja)  sim=0.99
  день  (ru)  sim=1.00
  day   (en)  sim=1.00
```

No dictionaries were used to produce these results.

---

## Architecture

```
Input: word rendered to 128x32 grayscale image
  ↓
CNN (4 layers, GroupNorm, ReLU, MaxPool)
  Conv(1→32) + GroupNorm(8,32) + ReLU + MaxPool
  Conv(32→64) + GroupNorm(8,64) + ReLU + MaxPool
  Conv(64→128) + GroupNorm(16,128) + ReLU + MaxPool
  Conv(128→256) + GroupNorm(16,256) + ReLU + AdaptiveAvgPool
  ↓
Projection head: FC(256) → ReLU → Dropout(0.1) → FC(256)
  ↓
L2 normalisation
  ↓
256-dimensional unit vector
```

Loss: NT-Xent (SimCLR), temperature=0.07. Each anchor sees 255 negatives per step at batch size 256.

---

## Training data

**Vocabulary:** Wikipedia word frequencies, 100,000 words per language, 10 languages, 1,000,000 words total.

Languages: Arabic, German, English, Spanish, French, Hindi, Japanese, Russian, Swedish, Chinese.

**Semantic concept pairs:** 249 manually curated concepts with translations across all 16 supported languages. These are the only labels in the entire pipeline. They define what counts as a positive pair across scripts, not what words mean.

**Pair construction:** 50% font-size variation pairs (same word, different sizes — teaches scale invariance), 50% cross-lingual concept pairs (same concept, different language — teaches script invariance).

---

## Adding new scripts from scanned images

The model can be extended with scanned word or glyph images — handwriting, historical documents, ancient scripts — without retraining the base model.

### Prepare data

Crop individual word or glyph images into a folder. No labels required.

```
data/
  my_script/
    images/
      0001.png
      0002.png
      ...
```

For Omniglot (50 scripts, ready to use):

```bash
python get_omniglot.py                      # list available scripts
python get_omniglot.py --alphabet Tifinagh  # prepare one script
```

### Add a script

```bash
python vwe.py add --name tifinagh --script tifinagh --images data/tifinagh/images/
```

This fine-tunes the base model on the new script with the early CNN layers frozen. The base model is not modified. The result is saved as `adapters/tifinagh.pt`.

### Search

Find the nearest vocabulary words for a scanned image:

```bash
python vwe.py search \
    --image data/tifinagh/images/character01_0910_01.png \
    --adapter tifinagh \
    --vocab vocabularies/

# Filter to specific languages
python vwe.py search \
    --image data/tifinagh/images/character01_0910_01.png \
    --adapter tifinagh \
    --vocab vocabularies/ --langs zh ja --sample 5000
```

### All commands

```bash
python vwe.py add        --name NAME --script SCRIPT --images DIR/
python vwe.py list
python vwe.py encode     --image FILE [--adapter NAME]
python vwe.py neighbours --image FILE --adapter NAME
python vwe.py search     --image FILE --adapter NAME --vocab vocabularies/
```

---

## What was fixed to get here

**Mode collapse (sim=1.00 for everything)**
The original contrastive loss used Euclidean distance with a margin of 1.0 on L2-normalised vectors. Maximum Euclidean distance between unit vectors is 2.0, so random word pairs were already beyond the margin. Only the positive loss contributed gradient. Everything collapsed to one point.

Fix: replaced with NT-Xent loss operating directly on cosine similarity. Every sample in the batch serves as a negative for every other sample. Dense gradient signal, no margin to miscalibrate.

**Train/eval discrepancy (collapse in production, not during training)**
BatchNorm computes running statistics during training. In eval mode it uses those statistics to normalise. Word images all share the same structure — white background, dark text — so the running mean converges to something that makes all words look identical in eval mode.

Fix: replaced BatchNorm2d with GroupNorm throughout. GroupNorm normalises per sample, behaviour is identical in train and eval.

**Vocabulary cap (model memorised 5k words, generalised to nothing)**
The dataset was loading only the top 5,000 words per language from each 100,000-word vocabulary. The model memorised which specific pairs to separate without learning transferable visual features.

Fix: removed the cap. Full vocabulary in training. Pairs no longer repeat, model is forced to generalise.

**Semantic concepts too few (10 → 249)**
With only 10 cross-lingual concept pairs the cross-script signal was drowned out by the font-variation signal. The model learned scale invariance but not script invariance.

Fix: expanded to 249 concepts covering nature, body, family, time, actions, objects, colours, numbers, governance, geography, religion, economy, and more.

---

## Known limitations

Complex CJK characters (high stroke count) collapse to similar vectors at 128×32 resolution. Single characters and short words work well. Dense multi-stroke characters are at the resolution ceiling of the current architecture.

---

## Usage

### Train

```bash
python3 visual_embeddings_torch.py
```

Resumes from `visual_embeddings.pt` if it exists.

### Build vocabulary files from Wikipedia

```bash
python3 build_vocabulary.py
```

Outputs one JSON file per language to `vocabularies/`.

### Embed a word in code

```python
from visual_embeddings_torch import load_model

model = load_model("visual_embeddings.pt")
vector = model.encode_word("水")   # returns numpy array, shape (256,)
```

---

## Requirements

```
torch
torchvision
pillow
numpy
tqdm
arabic-reshaper
python-bidi
```

```bash
pip install torch torchvision pillow numpy tqdm arabic-reshaper python-bidi
sudo apt install fonts-freefont-ttf fonts-noto-cjk
```

Trained on an RTX 2080. Runs on CPU as well, slower.

---

## Files

| File | Description |
|---|---|
| `visual_embeddings_torch.py` | Model, training loop, validation |
| `build_vocabulary.py` | Wikipedia vocabulary builder |
| `vwe.py` | Shell: add scripts, search, encode |
| `scan_dataset.py` | Dataset loader for scanned images |
| `finetune_scan.py` | Fine-tuning on scan data |
| `get_omniglot.py` | Download and prepare Omniglot scripts |

---

## License

Apache 2.0
