"""
Wikipedia Vocabulary Builder
Laddar ner Wikipedia för valfria språk och extraherar de vanligaste orden.
Sparar en vocab-fil per språk som visual_embeddings_torch.py kan läsa.

Run:  python3 build_vocabulary.py
      python3 build_vocabulary.py --langs en sv de fr --top 50000

Requirements:
    pip install datasets wikipedia-api tqdm
"""

import os
import re
import json
import argparse
import unicodedata
from collections import Counter
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────

LANGUAGES = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "sv": "swedish",
    "ru": "russian",
    "ar": "arabic",
    "hi": "hindi",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "el": "greek",
    "he": "hebrew",
    "th": "thai",
    "tr": "turkish",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
}

WIKI_DATE = "20231101"
TOP_N        = 50000
MIN_LEN      = 2
MAX_LEN      = 24
OUTPUT_DIR   = "vocabularies"

# ─── CLEANING ────────────────────────────────────────────────────────────────

# Characters to strip from word boundaries
STRIP_CHARS = '.,!?;:"\'''„"»«()[]{}|–—<>/\\*&#@^~`+=_\n\t'

# Regex: keep only words that are "real" words
# Allows Unicode letters/marks (covers all scripts), hyphens inside words
WORD_RE = re.compile(r"[\w][\w\-']*[\w]|[\w]", re.UNICODE)

def is_valid_word(word, lang):
    """
    Filter out noise: numbers, URLs, wiki markup, single chars.
    Language-aware minimum frequency is handled by Counter.
    """
    if len(word) < MIN_LEN or len(word) > MAX_LEN:
        return False
    # Skip pure numbers
    if word.isdigit():
        return False
    # Skip wiki markup remnants
    if any(c in word for c in ['=', '{', '}', '[', ']', '|', '<', '>']):
        return False
    # Skip URLs
    if 'http' in word or 'www.' in word:
        return False
    # Must contain at least one Unicode letter
    if not any(unicodedata.category(c).startswith('L') for c in word):
        return False
    return True

def clean_text(text, lang):
    """Extract valid words from a Wikipedia article text."""
    words = []
    for match in WORD_RE.finditer(text):
        word = match.group()
        # For non-CJK languages: lowercase
        if lang not in ("zh", "ja", "ko", "th", "ar", "he"):
            word = word.lower()
        if is_valid_word(word, lang):
            words.append(word)
    return words

# ─── CJK TOKENISATION ────────────────────────────────────────────────────────

def tokenize_cjk(text):
    """
    For Chinese/Japanese/Korean: split by character.
    Each CJK character is its own token.
    For Japanese we also keep katakana/hiragana runs.
    """
    tokens = []
    run = ""
    for ch in text:
        cp = ord(ch)
        is_cjk = (
            0x4E00 <= cp <= 0x9FFF or    # CJK Unified
            0x3400 <= cp <= 0x4DBF or    # CJK Extension A
            0x20000 <= cp <= 0x2A6DF or  # CJK Extension B
            0xF900 <= cp <= 0xFAFF       # CJK Compatibility
        )
        is_kana = (
            0x3040 <= cp <= 0x30FF or    # Hiragana + Katakana
            0xAC00 <= cp <= 0xD7A3       # Korean Hangul
        )
        if is_cjk:
            if run:
                tokens.append(run)
                run = ""
            tokens.append(ch)
        elif is_kana:
            run += ch
        else:
            if run:
                tokens.append(run)
                run = ""
    if run:
        tokens.append(run)
    return [t for t in tokens if len(t) >= 1]

# ─── DOWNLOAD & PROCESS ──────────────────────────────────────────────────────

def process_language(lang, top_n=TOP_N):
    """
    Download Wikipedia for one language, count word frequencies,
    return top_n words as a list.
    """
    from datasets import load_dataset

    lang_name = LANGUAGES.get(lang, lang)
    dataset_name = f"{WIKI_DATE}.{lang}"
    print(f"\n{'='*50}")
    print(f"Language: {lang_name} ({lang})")
    print(f"Dataset:  wikipedia / {dataset_name}")
    print(f"{'='*50}")

    try:
        dataset = load_dataset(
    "wikimedia/wikipedia",
    dataset_name,
    split="train",
)
        
    except Exception as e:
        print(f"  ERROR loading {lang}: {e}")
        print(f"  Skipping {lang}.")
        return []

    counter = Counter()
    is_cjk_lang = lang in ("zh", "ja", "ko")

    print(f"  Articles: {len(dataset):,}")
    print(f"  Processing...")

    for i, article in enumerate(tqdm(dataset, desc=f"  {lang}", leave=False)):
        text = article.get("text", "")
        if not text:
            continue

        if is_cjk_lang:
            words = tokenize_cjk(text)
        else:
            words = clean_text(text, lang)

        counter.update(words)

        # Progress checkpoint every 100k articles
        if (i + 1) % 100000 == 0:
            print(f"  {i+1:,} articles, {len(counter):,} unique tokens so far")

    print(f"  Total unique tokens: {len(counter):,}")
    print(f"  Total token count:   {sum(counter.values()):,}")

    # Take top N
    top_words = [word for word, count in counter.most_common(top_n)
                 if is_valid_word(word, lang) or lang in ("zh", "ja", "ko", "th")]

    top_words = top_words[:top_n]
    print(f"  Top {len(top_words):,} words selected")

    # Show sample
    sample = top_words[100:115]  # skip stopwords at top
    print(f"  Sample (rank 100-115): {sample}")

    return top_words

# ─── SAVE ────────────────────────────────────────────────────────────────────

def save_vocabulary(lang, words, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    lang_name = LANGUAGES.get(lang, lang)
    path = os.path.join(output_dir, f"{lang}.json")
    data = {
        "language_code": lang,
        "language_name": lang_name,
        "word_count": len(words),
        "words": words,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path}  ({len(words):,} words)")
    return path

def load_vocabulary(lang, vocab_dir=OUTPUT_DIR):
    path = os.path.join(vocab_dir, f"{lang}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["words"]

def load_all_vocabularies(vocab_dir=OUTPUT_DIR):
    """Load all saved vocabularies into a dict {lang: [words]}."""
    result = {}
    if not os.path.exists(vocab_dir):
        return result
    for fname in os.listdir(vocab_dir):
        if fname.endswith(".json"):
            lang = fname[:-5]
            words = load_vocabulary(lang, vocab_dir)
            if words:
                result[lang] = words
                print(f"  Loaded {lang}: {len(words):,} words")
    return result

# ─── PAIR BUILDER ────────────────────────────────────────────────────────────

def build_cross_lingual_pairs(vocab_dict, n_pairs=50000):
    """
    Build training pairs from full Wikipedia vocabularies.
    
    Positive pairs:
      - Same word, different font sizes
      - Cognates: words sharing 4+ character prefix across related languages
        (night/Nacht, water/Wasser, star/Stern etc.)
    
    Negative pairs:
      - Random words from different languages
      - Random words within same language
    """
    langs = list(vocab_dict.keys())
    all_words_by_lang = vocab_dict

    # Latin-script language groups for cognate detection
    LATIN_LANGS = {"en","de","fr","es","it","sv","nl","pl","pt","tr"}

    def likely_cognate(w1, w2):
        """
        Rough cognate heuristic:
        Share 4+ char prefix after normalising to ASCII approximation.
        Not precise — but good enough for training signal.
        """
        if len(w1) < 4 or len(w2) < 4:
            return False
        # Normalise: remove diacritics
        def strip_diacritics(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
            ).lower()
        n1 = strip_diacritics(w1)
        n2 = strip_diacritics(w2)
        # Check shared prefix of 4+ chars
        shared = 0
        for a, b in zip(n1, n2):
            if a == b:
                shared += 1
            else:
                break
        return shared >= 4

    pairs = []
    n_per_type = n_pairs // 3

    # Type 1: same word, font size variation
    print(f"  Building {n_per_type:,} font-variation pairs...")
    all_words = [(lang, w) for lang, words in all_words_by_lang.items()
                 for w in words[:5000]]  # use top 5000 per lang
    for _ in range(n_per_type):
        lang, word = random.choice(all_words)
        pairs.append({
            "word_a": word, "size_a": random.choice([17,18,19,20,21,22,23]),
            "word_b": word, "size_b": random.choice([17,18,19,20,21,22,23]),
            "label": 1
        })

    # Type 2: cognates across latin-script languages
    print(f"  Building {n_per_type:,} cognate pairs...")
    latin_words = {lang: words[:3000] for lang, words in all_words_by_lang.items()
                   if lang in LATIN_LANGS}
    latin_lang_list = list(latin_words.keys())

    cognate_count = 0
    attempts = 0
    while cognate_count < n_per_type and attempts < n_per_type * 20:
        attempts += 1
        if len(latin_lang_list) < 2:
            break
        l1, l2 = random.sample(latin_lang_list, 2)
        w1 = random.choice(latin_words[l1])
        w2 = random.choice(latin_words[l2])
        if likely_cognate(w1, w2):
            pairs.append({
                "word_a": w1, "size_a": 20,
                "word_b": w2, "size_b": 20,
                "label": 1
            })
            cognate_count += 1

    # Fill remaining positives with random same-lang pairs
    remaining = n_per_type - cognate_count
    for _ in range(remaining):
        lang = random.choice(langs)
        w1 = random.choice(all_words_by_lang[lang][:5000])
        w2 = random.choice(all_words_by_lang[lang][:5000])
        pairs.append({
            "word_a": w1, "size_a": 20,
            "word_b": w2, "size_b": 20,
            "label": 1 if w1 == w2 else 0
        })

    print(f"  Cognate pairs found: {cognate_count:,} / {n_per_type:,} attempted")

    # Type 3: negatives
    print(f"  Building {n_per_type:,} negative pairs...")
    for _ in range(n_per_type):
        l1, l2 = random.sample(langs, 2)
        w1 = random.choice(all_words_by_lang[l1][:5000])
        w2 = random.choice(all_words_by_lang[l2][:5000])
        pairs.append({
            "word_a": w1, "size_a": 20,
            "word_b": w2, "size_b": 20,
            "label": 0
        })

    import random as rnd
    rnd.shuffle(pairs)
    print(f"  Total pairs: {len(pairs):,}")
    return pairs

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    import random

    parser = argparse.ArgumentParser(description="Build Wikipedia vocabularies")
    parser.add_argument("--langs", nargs="+",
                        default=["en","sv","de","fr","es","ru","ar","hi","zh","ja"],
                        help="Language codes to download")
    parser.add_argument("--top", type=int, default=TOP_N,
                        help="Words per language")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip languages already downloaded")
    args = parser.parse_args()

    print("=" * 60)
    print("WIKIPEDIA VOCABULARY BUILDER")
    print("=" * 60)
    print(f"Languages: {args.langs}")
    print(f"Words/lang: {args.top:,}")
    print(f"Output dir: {args.output}")
    print()
    print("Disk estimate:")
    for lang in args.langs:
        gb = {"zh": 18, "en": 21, "de": 19, "fr": 17, "ru": 15,
              "ar": 8, "ja": 9, "hi": 5, "es": 14, "sv": 4}.get(lang, 6)
        print(f"  {lang}: ~{gb}GB download, ~{args.top//1000}MB vocab file")
    print()
    print("Tip: start with --langs en sv de to test the pipeline first.")
    print()

    for lang in args.langs:
        vocab_path = os.path.join(args.output, f"{lang}.json")

        if args.skip_existing and os.path.exists(vocab_path):
            print(f"Skipping {lang} (already exists: {vocab_path})")
            continue

        words = process_language(lang, top_n=args.top)
        if words:
            save_vocabulary(lang, words, args.output)

    print("\n" + "=" * 60)
    print("Done. Vocabularies saved to:", args.output)
    print()
    print("To use in training:")
    print("  from build_vocabulary import load_all_vocabularies")
    print("  vocab = load_all_vocabularies('vocabularies')")
    print("  pairs = build_cross_lingual_pairs(vocab, n_pairs=100000)")
    print("=" * 60)

if __name__ == "__main__":
    import random
    main()
