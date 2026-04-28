"""
Visual Word Embeddings — PyTorch GPU version
Cross-lingual, script-agnostic visual embeddings

Run:  python3 visual_embeddings_torch.py

Requirements:
    pip install torch torchvision pillow numpy arabic-reshaper python-bidi
    sudo apt install fonts-freefont-ttf fonts-noto-cjk
"""

import numpy as np
import random
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────

IMG_W       = 128
IMG_H       = 32
EMBED_DIM   = 256
BATCH_SIZE  = 128
LR          = 3e-4
EPOCHS      = 50
MARGIN      = 1.0
FONT_SIZE   = 20
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name(0)}")

# ─── ARABIC RESHAPING (optional) ─────────────────────────────────────────────

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_OK = True
    print("Arabic reshaping: enabled")
except ImportError:
    ARABIC_OK = False
    print("Arabic reshaping: disabled (pip install arabic-reshaper python-bidi)")

def reshape_arabic(word):
    """Reshape Arabic text for correct visual rendering."""
    if not ARABIC_OK:
        return word
    # Detect if word contains Arabic
    for ch in word:
        cp = ord(ch)
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            reshaped = arabic_reshaper.reshape(word)
            return get_display(reshaped)
    return word

# ─── FONT AUTO-DETECTION ─────────────────────────────────────────────────────

def _fc_match(name):
    try:
        r = subprocess.run(["fc-match", "-f", "%{file}", name],
                           capture_output=True, text=True, timeout=3)
        p = r.stdout.strip()
        if r.returncode == 0 and p and os.path.exists(p):
            return p
    except Exception:
        pass
    return None

def find_font(candidates, fc_names=None):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    if fc_names:
        for name in fc_names:
            p = _fc_match(name)
            if p:
                return p
    return None

FONT_PATH = find_font([
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/mnt/c/Windows/Fonts/arial.ttf",
], fc_names=["FreeSans", "DejaVu Sans", "Liberation Sans", "Arial"])

CJK_FONT_PATH = find_font([
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/mnt/c/Windows/Fonts/msyh.ttc",
], fc_names=["Noto Sans CJK", "PingFang SC", "MS Gothic"])

if FONT_PATH is None:
    raise RuntimeError(
        "No font found.\n"
        "  sudo apt install fonts-freefont-ttf\n"
        "  sudo apt install fonts-dejavu-core\n"
    )
if CJK_FONT_PATH is None:
    print("WARNING: No CJK font. Install: sudo apt install fonts-noto-cjk")
    CJK_FONT_PATH = FONT_PATH

print(f"Main font:  {FONT_PATH}")
print(f"CJK font:   {CJK_FONT_PATH}")

# ─── RENDERING ───────────────────────────────────────────────────────────────

def is_cjk(word):
    for ch in word:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or
            0x3040 <= cp <= 0x30FF or
            0xAC00 <= cp <= 0xD7A3):
            return True
    return False

def word_to_tensor(word, font_size=FONT_SIZE):
    """
    Render word to image tensor.
    Returns: (1, H, W) float32 tensor, 0=black, 1=white.
    """
    word = reshape_arabic(word)
    path = CJK_FONT_PATH if is_cjk(word) else FONT_PATH
    font = ImageFont.truetype(path, font_size)

    img = Image.new("L", (IMG_W, IMG_H), color=255)
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
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

# ─── ENCODER ─────────────────────────────────────────────────────────────────

class VisualWordEncoder(nn.Module):
    """
    CNN encoder for visual word embeddings.

    Architecture:
        Conv(1→32, k=3) → BN → ReLU → MaxPool(2)    [32 x 16 x 64]
        Conv(32→64, k=3) → BN → ReLU → MaxPool(2)   [64 x 8 x 32]
        Conv(64→128, k=3) → BN → ReLU → MaxPool(2)  [128 x 4 x 16]
        Conv(128→256, k=3) → BN → ReLU → AdaptAvgPool(1x1) [256]
        FC(256→256) → ReLU → FC(256→embed_dim) → L2 norm
    """
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
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
        """x: (B, 1, H, W) → (B, embed_dim) L2-normalised"""
        features = self.cnn(x)
        emb = self.projection(features)
        return F.normalize(emb, p=2, dim=1)

    def encode_word(self, word):
        """Encode a single word string → numpy vector."""
        self.eval()
        with torch.no_grad():
            t = word_to_tensor(word).unsqueeze(0).to(DEVICE)
            return self(t).squeeze(0).cpu().numpy()

# ─── CONTRASTIVE LOSS ────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    Standard contrastive loss (Hadsell et al. 2006).
    label=1: same class (minimise distance)
    label=0: different class (push apart up to margin)
    """
    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, labels):
        dist = F.pairwise_distance(emb_a, emb_b)
        pos_loss = labels * dist.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return (pos_loss + neg_loss).mean()

# ─── DATA ────────────────────────────────────────────────────────────────────

MULTILINGUAL_WORDS = {
    "english":  ["hello","world","run","running","night","light","water",
                 "fire","tree","house","love","time","hand","eye","day","sun",
                 "book","door","road","bird","stone","river","moon","star"],
    "german":   ["Nacht","Licht","Wasser","Feuer","Baum","Haus","Liebe",
                 "Zeit","Hand","Auge","Tag","Sonne","Buch","Mond","Stern"],
    "french":   ["nuit","lumière","eau","feu","arbre","maison","amour",
                 "temps","main","oeil","jour","soleil","livre","lune","étoile"],
    "spanish":  ["noche","luz","agua","fuego","árbol","casa","amor",
                 "tiempo","mano","ojo","día","sol","libro","luna","estrella"],
    "italian":  ["notte","luce","acqua","fuoco","albero","casa","amore",
                 "tempo","mano","occhio","giorno","sole","libro","luna","stella"],
    "russian":  ["ночь","свет","вода","огонь","дерево","дом","любовь",
                 "время","рука","глаз","день","солнце","книга","луна","звезда"],
    "arabic":   ["ليل","ضوء","ماء","نار","شجرة","بيت","حب",
                 "وقت","يد","عين","يوم","شمس","كتاب","قمر","نجمة"],
    "hindi":    ["रात","प्रकाश","पानी","आग","पेड़","घर","प्यार",
                 "समय","हाथ","आँख","दिन","सूरज","किताब","चाँद","तारा"],
    "chinese":  ["夜","光","水","火","树","家","爱","时间","手","眼","天","太阳","书","月","星"],
    "japanese": ["夜","光","水","火","木","家","愛","時間","手","目","日","太陽","本","月","星"],
    "korean":   ["밤","빛","물","불","나무","집","사랑","시간","손","눈","날","태양","책","달","별"],
    "greek":    ["νύχτα","φως","νερό","φωτιά","δέντρο","σπίτι","αγάπη",
                 "χρόνος","χέρι","μάτι","μέρα","ήλιος","βιβλίο","φεγγάρι","αστέρι"],
    "hebrew":   ["לילה","אור","מים","אש","עץ","בית","אהבה",
                 "זמן","יד","עין","יום","שמש","ספר","ירח","כוכב"],
    "thai":     ["กลางคืน","แสง","น้ำ","ไฟ","ต้นไม้","บ้าน","ความรัก",
                 "เวลา","มือ","ตา","วัน","ดวงอาทิตย์","หนังสือ","ดวงจันทร์","ดาว"],
    "turkish":  ["gece","ışık","su","ateş","ağaç","ev","aşk",
                 "zaman","el","göz","gün","güneş","kitap","ay","yıldız"],
    "swedish":  ["natt","ljus","vatten","eld","träd","hus","kärlek",
                 "tid","hand","öga","dag","sol","bok","måne","stjärna"],
}

SEMANTIC_CONCEPTS = [
    ("night", ["night","Nacht","nuit","noche","notte","ночь","ليل","रात","夜","νύχτα","לילה","กลางคืน","gece","natt"]),
    ("water", ["water","Wasser","eau","agua","acqua","вода","ماء","पानी","水","νερό","מים","น้ำ","su","vatten"]),
    ("fire",  ["fire","Feuer","feu","fuego","fuoco","огонь","نار","आग","火","φωτιά","אש","ไฟ","ateş","eld"]),
    ("house", ["house","Haus","maison","casa","casa","дом","بيت","घर","家","σπίτι","בית","บ้าน","ev","hus"]),
    ("love",  ["love","Liebe","amour","amor","amore","любовь","حب","प्यार","爱","αγάπη","אהבה","ความรัก","aşk","kärlek"]),
    ("hand",  ["hand","Hand","main","mano","mano","рука","يد","हाथ","手","χέρι","יד","มือ","el","hand"]),
    ("eye",   ["eye","Auge","oeil","ojo","occhio","глаз","عين","आँख","眼","μάτι","עין","ตา","göz","öga"]),
    ("sun",   ["sun","Sonne","soleil","sol","sole","солнце","شمس","सूरज","太阳","ήλιος","שמש","ดวงอาทิตย์","güneş","sol"]),
    ("moon",  ["moon","Mond","lune","luna","luna","луна","قمر","चाँद","月","φεγγάρι","ירח","ดวงจันทร์","ay","måne"]),
    ("book",  ["book","Buch","livre","libro","libro","книга","كتاب","किताब","书","βιβλίο","ספר","หนังสือ","kitap","bok"]),
]


# ─── WIKIPEDIA VOCABULARY LOADER ─────────────────────────────────────────────

import json

def load_wiki_vocab(vocab_dir="vocabularies"):
    """Load all saved Wikipedia vocabularies from build_vocabulary.py."""
    result = {}
    if not os.path.exists(vocab_dir):
        print(f"  No vocabularies dir found at '{vocab_dir}' — using built-in words")
        return result
    for fname in sorted(os.listdir(vocab_dir)):
        if fname.endswith(".json"):
            lang = fname[:-5]
            try:
                with open(os.path.join(vocab_dir, fname), encoding="utf-8") as f:
                    data = json.load(f)
                result[lang] = data["words"]
                print(f"  Loaded {lang:4}: {len(data['words']):>6,} words")
            except Exception as e:
                print(f"  WARN: could not load {fname}: {e}")
    return result

class WordPairDataset(Dataset):
    """
    Three types of pairs:
    1. Same word, font size variation          → label=1 (strong positive)
    2. Same concept across languages           → label=1 (semantic positive)
    3. Different random words                  → label=0 (negative)
    """
    def __init__(self, n_pairs=5000, vocab_dir="vocabularies"):
        self.pairs = []

        # Load vocab
        wiki_vocab = load_wiki_vocab(vocab_dir)
        if wiki_vocab:
            word_list = [w for words in wiki_vocab.values() for w in words[:5000]]
            print(f"  Wikipedia vocab: {len(wiki_vocab)} languages, {len(word_list):,} words")
        else:
            word_list = [w for ws in MULTILINGUAL_WORDS.values() for w in ws]
            print(f"  Built-in vocab: {len(word_list)} words")

        # Build raw string pairs first
        raw_pairs = []
        for _ in range(n_pairs // 3):
            word = random.choice(word_list)
            sz_a = random.choice([17, 18, 19, 20, 21, 22, 23])
            sz_b = random.choice([17, 18, 19, 20, 21, 22, 23])
            raw_pairs.append((word, sz_a, word, sz_b, 1.0))
        for _ in range(n_pairs // 3):
            _, translations = random.choice(SEMANTIC_CONCEPTS)
            a = random.choice(translations)
            b = random.choice(translations)
            raw_pairs.append((a, FONT_SIZE, b, FONT_SIZE, 1.0))
        for _ in range(n_pairs // 3):
            a = random.choice(word_list)
            b = random.choice(word_list)
            while b == a:
                b = random.choice(word_list)
            raw_pairs.append((a, FONT_SIZE, b, FONT_SIZE, 0.0))
        random.shuffle(raw_pairs)

        # Pre-render ALL images into RAM — GPU never waits for CPU during training
        print(f"  Pre-rendering {len(raw_pairs):,} pairs into RAM...")
        for wa, sa, wb, sb, label in tqdm(raw_pairs, desc="  rendering"):
            ta = word_to_tensor(wa, sa)
            tb = word_to_tensor(wb, sb)
            self.pairs.append((ta, tb, torch.tensor(label, dtype=torch.float32)))
        print(f"  Pre-rendering done.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ta, tb, label = self.pairs[idx]
        return ta, tb, label

# ─── TRAINING ────────────────────────────────────────────────────────────────

def train(model, epochs=EPOCHS, n_pairs=5000, lr=LR):
    print(f"\n=== TRAINING on {DEVICE} ===")
    print(f"  Pairs: {n_pairs}  Batch: {BATCH_SIZE}  Epochs: {epochs}")

    dataset = WordPairDataset(n_pairs=n_pairs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=True)

    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for ta, tb, labels in loader:
            ta = ta.to(DEVICE)
            tb = tb.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            emb_a = model(ta)
            emb_b = model(tb)
            loss = criterion(emb_a, emb_b, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}  lr={lr_now:.2e}")

    return model

# ─── VALIDATION ──────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def validate_similarity(model, label=""):
    if label:
        print(f"\n=== SIMILARITY ({label}) ===")
    tests = [
        ("same word font var",     "water",  "water",   "high"),
        ("en/de same concept",     "water",  "Wasser",  "medium"),
        ("en/zh same concept",     "fire",   "火",      "medium"),
        ("en/ar same concept",     "house",  "بيت",     "medium"),
        ("en/ru same concept",     "moon",   "луна",    "medium"),
        ("en/sw same concept",     "love",   "kärlek",  "medium"),
        ("prefix run/running",     "run",    "running", "medium"),
        ("unrelated en",           "hello",  "fire",    "low"),
        ("unrelated cross-script", "house",  "夜",      "low"),
        ("unrelated en/de",        "love",   "Baum",    "low"),
    ]
    sims = {"high": [], "medium": [], "low": []}
    for desc, wa, wb, exp in tests:
        ea = model.encode_word(wa)
        eb = model.encode_word(wb)
        sim = cosine_sim(ea, eb)
        sims[exp].append(sim)
        print(f"  {exp:6}  sim={sim:+.3f}  {desc}")

    avg_h = np.mean(sims["high"])
    avg_m = np.mean(sims["medium"])
    avg_l = np.mean(sims["low"])
    ok = avg_h >= avg_m >= avg_l
    print(f"  Averages: high={avg_h:.3f}  medium={avg_m:.3f}  low={avg_l:.3f}")
    print(f"  Ordering: {'PASS' if ok else 'NOT YET'}")
    return ok

def validate_clustering(model):
    print("\n=== SCRIPT CLUSTERING ===")
    scripts = {
        "latin":      ["hello", "world", "water", "fire", "house", "love"],
        "cyrillic":   ["ночь",  "вода",  "огонь", "дом",  "рука",  "луна"],
        "arabic":     ["ليل",   "ماء",   "نار",   "بيت",  "يد",    "قمر"],
        "cjk":        ["夜",    "水",    "火",    "家",   "手",    "月"],
        "devanagari": ["रात",   "पानी",  "आग",   "घर",   "हाथ",  "चाँद"],
        "thai":       ["กลางคืน","น้ำ", "ไฟ",   "บ้าน", "มือ",  "ดวงจันทร์"],
    }
    embs = {s: [model.encode_word(w) for w in ws] for s, ws in scripts.items()}
    results = []
    for s1, e1 in embs.items():
        within = np.mean([cosine_sim(e1[i], e1[j])
                          for i in range(len(e1)) for j in range(i+1, len(e1))])
        between = np.mean([cosine_sim(a, b)
                           for s2, e2 in embs.items() if s2 != s1
                           for a in e1 for b in e2])
        ok = within > between
        results.append(ok)
        print(f"  {'OK  ' if ok else 'FAIL'}  {s1:12}  within={within:+.3f}  between={between:+.3f}")
    return all(results)

def nearest_neighbours(model, query_word, word_pool, n=5):
    """Find N most visually similar words to query."""
    q_emb = model.encode_word(query_word)
    scored = []
    for word in word_pool:
        emb = model.encode_word(word)
        scored.append((cosine_sim(q_emb, emb), word))
    scored.sort(reverse=True)
    return scored[:n]

def demo_nearest_neighbours(model, vocab_dir="vocabularies"):
    print("\n=== NEAREST NEIGHBOURS (visual similarity) ===")
    wiki_vocab = load_wiki_vocab(vocab_dir)
    if wiki_vocab:
        # Sample 500 words per language for speed
        all_words = [w for words in wiki_vocab.values() for w in words[:500]]
    else:
        all_words = [w for ws in MULTILINGUAL_WORDS.values() for w in ws]
    queries = ["water", "fire", "love", "手", "بيت", "natt", "Wasser"]
    for q in queries:
        neighbours = nearest_neighbours(model, q, all_words, n=6)
        nn_str = "  ".join(f"{w}({s:.2f})" for s, w in neighbours[1:])
        print(f"  '{q}'  →  {nn_str}")

# ─── SAVE / LOAD ─────────────────────────────────────────────────────────────

def save_model(model, path="visual_embeddings.pt"):
    torch.save({
        "model_state": model.state_dict(),
        "embed_dim": model.projection[-1].out_features,
        "img_w": IMG_W,
        "img_h": IMG_H,
    }, path)
    print(f"Model saved: {path}")

def load_model(path="visual_embeddings.pt"):
    ckpt = torch.load(path, map_location=DEVICE)
    model = VisualWordEncoder(embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded: {path}")
    return model

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("VISUAL WORD EMBEDDINGS — PyTorch GPU version")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Rendering smoke test
    print("\n=== RENDERING TEST ===")
    test_words = [
        ("english","hello"), ("arabic","بيت"), ("hindi","पानी"),
        ("chinese","水"), ("russian","вода"), ("thai","น้ำ"),
        ("korean","달"), ("hebrew","אור"), ("greek","νερό"),
    ]
    for lang, word in test_words:
        t = word_to_tensor(word)
        ink = int((t < 0.9).sum())
        print(f"  {'OK  ' if ink > 5 else 'FAIL'}  {lang:10}  '{word}'  ink={ink}")

    # Build model
    print("\n=== MODEL ===")
    model = VisualWordEncoder(embed_dim=EMBED_DIM).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Embed dim:  {EMBED_DIM}")

    # Baseline
    validate_similarity(model, "before training")

    # Train
    model = train(model, epochs=EPOCHS, n_pairs=50000, lr=LR)
    model.eval()

    # Evaluate
    validate_similarity(model, "after training")
    validate_clustering(model)
    demo_nearest_neighbours(model)

    # Save
    save_model(model, "visual_embeddings.pt")

    print("\n" + "=" * 60)
    print("Done.")
    print("Load later with:  model = load_model('visual_embeddings.pt')")
    print("Embed a word with: model.encode_word('hello')")
    print("=" * 60)

if __name__ == "__main__":
    main()
