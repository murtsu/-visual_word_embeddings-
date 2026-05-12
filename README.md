# visual_word_embeddings

Cross-lingual, script-agnostic word embeddings trained on visual appearance alone.

No tokenizer. No dictionary. No pretrained vectors. No labels. No CUDA out of memory errors — well, fewer.

Type `water` and the model finds `agua`, `水`, and `su`. It does not know those words mean the same thing. It figured it out by looking at the shapes. Which is more than can be said for most transformer models that needed the entire internet to reach the same conclusion.

---

## How it works

Every word gets rendered to a 384×96 grayscale image. A CNN reads the image and produces a 256-dimensional vector. Words that look visually similar, or that the model has learned to associate through cross-lingual training pairs, end up close together in that vector space.

The model never sees text as text. It sees pixels.

This is either a breakthrough or a very expensive way to do OCR, depending on who you ask.

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

No dictionaries were used. The model has never read a sentence. It has also never complained about its training data, which puts it ahead of several colleagues.

---

## Architecture

```
Input: word rendered to 384×96 grayscale image
  ↓
CNN encoder
  Conv(1→32)   + GroupNorm(8,32)   + ReLU + MaxPool(2)
  Conv(32→64)  + GroupNorm(8,64)   + ReLU + MaxPool(2)
  Conv(64→128) + GroupNorm(8,128)  + ReLU + MaxPool(2)
  Conv(128→256)+ GroupNorm(8,256)  + ReLU + AdaptiveAvgPool(1×1)
  ↓
Projection head
  FC(256) → ReLU → Dropout(0.1) → FC(256)
  ↓
L2 normalisation
  ↓
256-dimensional unit vector
```

GroupNorm throughout. Not BatchNorm. BatchNorm was tried. BatchNorm had opinions about running statistics in eval mode that did not align with producing useful embeddings. GroupNorm does not have opinions. It normalises per sample and goes home.

Loss: NT-Xent (SimCLR), temperature=0.07. Every sample in the batch serves as a negative for every other sample. Dense gradient signal, no margin to miscalibrate, considerably fewer existential crises than the previous contrastive loss.

---

## Training data

Wikipedia word frequencies, 100,000 words per language, 10 languages. Languages: Arabic, German, English, Spanish, French, Hindi, Japanese, Russian, Swedish, Chinese.

144 manually curated semantic concepts with translations across all languages. These are the only labels in the pipeline. They define what counts as a positive pair across scripts — not what words mean, not what language sounds like, not anything a linguist spent years learning. Just: these shapes belong together.

Pair construction: 20% font-size variation (same word, different sizes — teaches scale invariance), 80% cross-lingual concept pairs (same concept, different script — teaches everything else). This ratio was found through experimentation. The previous ratio taught the model to match visual similarity but forget that Wasser and 水 are related. 80% is not a round number that was chosen for elegance.

---

## Langpaks and adapters

The model supports a modular extension system. The base model handles the 10 training languages. New scripts are added as adapter files without retraining the base.

**Langpak:** A specialized model trained on a subset of languages. Used as an embedding anchor for scripts that have no dictionary. Think of it as a neighbourhood watch for a specific family of writing systems.

**Adapter:** A fine-tuned model for a specific script. Trained with the base model's early layers frozen. The early layers detect edges and strokes — these are universal. The later layers learn what to do with them — these are script-specific.

The files are `.pt` and can be combined freely. One base, several langpaks, many adapters. Lego.

---

## Adding new scripts

The model accepts scanned word or glyph images — handwriting, historical documents, ancient inscriptions, things that predate Unicode by several centuries.

**Prepare data.** Crop individual glyphs into a folder. No labels required. No ground truth. No annotations. Just images.

```
data/
  my_script/
    images/
      0001.png
      0002.png
```

**For Omniglot** (50 writing systems, some invented, all hand-drawn, free):

```bash
python get_omniglot.py                      # list available scripts
python get_omniglot.py --alphabet Tifinagh  # prepare one
python get_omniglot.py --all                # prepare all 30 background scripts
```

**Add a script:**

```bash
python vwe.py add --name tifinagh --script tifinagh --images data/tifinagh/images/
```

**Search:**

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

**Search across multiple models:**

```bash
python vwe.py multi-search \
    --image data/tifinagh/images/character04_0913_03.png \
    --adapters tifinagh \
    --langpaks langpaks/asian.pt \
    --vocab vocabularies/ --langs zh ja
```

**All commands:**

```bash
python vwe.py add          --name NAME --script SCRIPT --images DIR/
python vwe.py list
python vwe.py encode       --image FILE [--adapter NAME]
python vwe.py neighbours   --image FILE --adapter NAME
python vwe.py search       --image FILE --adapter NAME --vocab vocabularies/
python vwe.py multi-search --image FILE --adapters NAME [NAME ...] --langpaks FILE [FILE ...]
```

---

## What was fixed to get here

**Mode collapse (sim=1.00 for everything, always)**
The original contrastive loss used Euclidean distance with a margin of 1.0 on L2-normalised vectors. Maximum Euclidean distance between unit vectors is 2.0, so random word pairs were already beyond the margin before training started. Only the positive loss contributed any gradient at all. Everything collapsed to a single point in embedding space, which is technically a valid geometric configuration and completely useless.

Fix: NT-Xent. Every sample in the batch is a negative for every other. No margin. No miscalibration. Loss starts at log(2×batch_size) and goes down from there.

**Train/eval discrepancy (works during training, collapses in production)**
BatchNorm computes running statistics across the training distribution. Word images all look roughly the same — white background, dark text, same aspect ratio — so the running mean converges to a value that normalises away all the signal. In eval mode, every word becomes the same vector. Impressive, in a way.

Fix: GroupNorm. Normalises per sample. Identical behaviour in train and eval. Does not accumulate opinions about your data distribution.

**Vocabulary cap (memorised 5k words, generalised to nothing)**
An early version loaded only the top 5,000 words per language. The model memorised which pairs to separate without learning anything transferable. It was, in the technical sense, overfitting. In the colloquial sense, it was cheating.

Fix: removed the cap. Full 100,000 words. Pairs do not repeat. The model is forced to generalise or fail visibly.

**Too few semantic concepts (10 concepts, 0 cross-script signal)**
With 10 cross-lingual concept pairs the semantic signal was drowned out by font-variation pairs. The model learned scale invariance (not entirely useless) and nothing else (entirely useless for the stated purpose).

Fix: 144 concepts across nature, body, family, time, actions, objects, colours, numbers, governance, and geography. The ratio shifted to 80% semantic pairs. Wasser found 水 again.

---

## Known limitations

Complex CJK characters with high stroke counts produce similar vectors at 384×96 resolution. A character that takes a calligrapher three seconds to write does not fit comfortably in a 384×96 grayscale image. This is not a model problem. It is a pixel budget problem. The fix is higher resolution and retraining the base model, which is on the list.

Simple CJK characters, Latin, Cyrillic, Arabic, Devanagari, and Thai work well. The boundary is approximately where stroke density starts exceeding what the canvas can represent. You will know when you find it.

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
sudo apt install fonts-noto fonts-noto-cjk
```

Trained on an RTX 2080 with 8GB VRAM. Runs on CPU at a speed that will test your relationship with waiting. The model does not require an internet connection, a subscription, or an account. It requires electricity and approximately 1.2 GB of disk space.

---

## Files

| File | Description |
|---|---|
| `visual_embeddings_torch.py` | Base model, training loop, validation |
| `build_vocabulary.py` | Wikipedia vocabulary builder |
| `build_langpak.py` | Langpak trainer for language subsets |
| `get_omniglot.py` | Omniglot downloader and preparer |
| `scan_dataset.py` | Dataset loader for scanned images |
| `finetune_scan.py` | Fine-tuning on scan data |
| `vwe.py` | Main interface |

---

## License

Apache 2.0
