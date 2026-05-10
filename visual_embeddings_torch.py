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
BATCH_SIZE  = 256
LR          = 3e-4
EPOCHS      = 100
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
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
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

class NTXentLoss(nn.Module):
    """
    NT-Xent loss (SimCLR). Each sample pairs with its counterpart;
    all other batch samples are negatives. batch_size=256 → 255 negatives per step.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a, emb_b):
        B = emb_a.size(0)
        z = torch.cat([emb_a, emb_b], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(B,        device=z.device),
        ])
        return F.cross_entropy(sim, labels)

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
    ('night'     , ['night', 'Nacht', 'nuit', 'noche', 'notte', 'ночь', 'ليل', 'रात', '夜', '夜', '밤', 'νύχτα', 'לילה', 'กลางคืน', 'gece', 'natt']),
    ('day'       , ['day', 'Tag', 'jour', 'día', 'giorno', 'день', 'يوم', 'दिन', '天', '日', '날', 'μέρα', 'יום', 'วัน', 'gün', 'dag']),
    ('morning'   , ['morning', 'Morgen', 'matin', 'mañana', 'mattina', 'утро', 'صباح', 'सुबह', '早晨', '朝', '아침', 'πρωί', 'בוקר', 'เช้า', 'sabah', 'morgon']),
    ('evening'   , ['evening', 'Abend', 'soir', 'tarde', 'sera', 'вечер', 'مساء', 'शाम', '傍晚', '夕方', '저녁', 'βράδυ', 'ערב', 'เย็น', 'akşam', 'kväll']),
    ('sun'       , ['sun', 'Sonne', 'soleil', 'sol', 'sole', 'солнце', 'شمس', 'सूरज', '太阳', '太陽', '태양', 'ήλιος', 'שמש', 'ดวงอาทิตย์', 'güneş', 'sol']),
    ('moon'      , ['moon', 'Mond', 'lune', 'luna', 'luna', 'луна', 'قمر', 'चाँद', '月', '月', '달', 'φεγγάρι', 'ירח', 'ดวงจันทร์', 'ay', 'måne']),
    ('star'      , ['star', 'Stern', 'étoile', 'estrella', 'stella', 'звезда', 'نجمة', 'तारा', '星', '星', '별', 'αστέρι', 'כוכב', 'ดาว', 'yıldız', 'stjärna']),
    ('sky'       , ['sky', 'Himmel', 'ciel', 'cielo', 'cielo', 'небо', 'سماء', 'आकाश', '天空', '空', '하늘', 'ουρανός', 'שמיים', 'ท้องฟ้า', 'gökyüzü', 'himmel']),
    ('cloud'     , ['cloud', 'Wolke', 'nuage', 'nube', 'nuvola', 'облако', 'سحابة', 'बादल', '云', '雲', '구름', 'σύννεφο', 'ענן', 'เมฆ', 'bulut', 'moln']),
    ('rain'      , ['rain', 'Regen', 'pluie', 'lluvia', 'pioggia', 'дождь', 'مطر', 'बारिश', '雨', '雨', '비', 'βροχή', 'גשם', 'ฝน', 'yağmur', 'regn']),
    ('snow'      , ['snow', 'Schnee', 'neige', 'nieve', 'neve', 'снег', 'ثلج', 'बर्फ', '雪', '雪', '눈', 'χιόνι', 'שלג', 'หิมะ', 'kar', 'snö']),
    ('wind'      , ['wind', 'Wind', 'vent', 'viento', 'vento', 'ветер', 'ريح', 'हवा', '风', '風', '바람', 'άνεμος', 'רוח', 'ลม', 'rüzgar', 'vind']),
    ('water'     , ['water', 'Wasser', 'eau', 'agua', 'acqua', 'вода', 'ماء', 'पानी', '水', '水', '물', 'νερό', 'מים', 'น้ำ', 'su', 'vatten']),
    ('fire'      , ['fire', 'Feuer', 'feu', 'fuego', 'fuoco', 'огонь', 'نار', 'आग', '火', '火', '불', 'φωτιά', 'אש', 'ไฟ', 'ateş', 'eld']),
    ('earth'     , ['earth', 'Erde', 'terre', 'tierra', 'terra', 'земля', 'أرض', 'पृथ्वी', '地球', '地球', '지구', 'γη', 'אדמה', 'โลก', 'toprak', 'jord']),
    ('sea'       , ['sea', 'Meer', 'mer', 'mar', 'mare', 'море', 'بحر', 'समुद्र', '海', '海', '바다', 'θάλασσα', 'ים', 'ทะเล', 'deniz', 'hav']),
    ('river'     , ['river', 'Fluss', 'rivière', 'río', 'fiume', 'река', 'نهر', 'नदी', '河', '川', '강', 'ποτάμι', 'נהר', 'แม่น้ำ', 'nehir', 'flod']),
    ('lake'      , ['lake', 'See', 'lac', 'lago', 'lago', 'озеро', 'بحيرة', 'झील', '湖', '湖', '호수', 'λίμνη', 'אגם', 'ทะเลสาบ', 'göl', 'sjö']),
    ('mountain'  , ['mountain', 'Berg', 'montagne', 'montaña', 'montagna', 'гора', 'جبل', 'पहाड़', '山', '山', '산', 'βουνό', 'הר', 'ภูเขา', 'dağ', 'berg']),
    ('forest'    , ['forest', 'Wald', 'forêt', 'bosque', 'foresta', 'лес', 'غابة', 'जंगल', '森林', '森', '숲', 'δάσος', 'יער', 'ป่า', 'orman', 'skog']),
    ('tree'      , ['tree', 'Baum', 'arbre', 'árbol', 'albero', 'дерево', 'شجرة', 'पेड़', '树', '木', '나무', 'δέντρο', 'עץ', 'ต้นไม้', 'ağaç', 'träd']),
    ('flower'    , ['flower', 'Blume', 'fleur', 'flor', 'fiore', 'цветок', 'زهرة', 'फूल', '花', '花', '꽃', 'λουλούδι', 'פרח', 'ดอกไม้', 'çiçek', 'blomma']),
    ('grass'     , ['grass', 'Gras', 'herbe', 'hierba', 'erba', 'трава', 'عشب', 'घास', '草', '草', '풀', 'χόρτο', 'עשב', 'หญ้า', 'çimen', 'gräs']),
    ('stone'     , ['stone', 'Stein', 'pierre', 'piedra', 'pietra', 'камень', 'حجر', 'पत्थर', '石', '石', '돌', 'πέτρα', 'אבן', 'หิน', 'taş', 'sten']),
    ('sand'      , ['sand', 'Sand', 'sable', 'arena', 'sabbia', 'песок', 'رمل', 'रेत', '沙', '砂', '모래', 'άμμος', 'חול', 'ทราย', 'kum', 'sand']),
    ('ice'       , ['ice', 'Eis', 'glace', 'hielo', 'ghiaccio', 'лёд', 'جليد', 'बर्फ', '冰', '氷', '얼음', 'πάγος', 'קרח', 'น้ำแข็ง', 'buz', 'is']),
    ('light'     , ['light', 'Licht', 'lumière', 'luz', 'luce', 'свет', 'ضوء', 'प्रकाश', '光', '光', '빛', 'φως', 'אור', 'แสง', 'ışık', 'ljus']),
    ('dark'      , ['dark', 'Dunkel', 'obscur', 'oscuro', 'buio', 'тьма', 'ظلام', 'अंधेरा', '黑暗', '暗', '어둠', 'σκοτάδι', 'חושך', 'ความมืด', 'karanlık', 'mörker']),
    ('shadow'    , ['shadow', 'Schatten', 'ombre', 'sombra', 'ombra', 'тень', 'ظل', 'छाया', '影子', '影', '그림자', 'σκιά', 'צל', 'เงา', 'gölge', 'skugga']),
    ('hand'      , ['hand', 'Hand', 'main', 'mano', 'mano', 'рука', 'يد', 'हाथ', '手', '手', '손', 'χέρι', 'יד', 'มือ', 'el', 'hand']),
    ('eye'       , ['eye', 'Auge', 'oeil', 'ojo', 'occhio', 'глаз', 'عين', 'आँख', '眼', '目', '눈', 'μάτι', 'עין', 'ตา', 'göz', 'öga']),
    ('ear'       , ['ear', 'Ohr', 'oreille', 'oreja', 'orecchio', 'ухо', 'أذن', 'कान', '耳', '耳', '귀', 'αυτί', 'אוזן', 'หู', 'kulak', 'öra']),
    ('nose'      , ['nose', 'Nase', 'nez', 'nariz', 'naso', 'нос', 'أنف', 'नाक', '鼻', '鼻', '코', 'μύτη', 'אף', 'จมูก', 'burun', 'näsa']),
    ('mouth'     , ['mouth', 'Mund', 'bouche', 'boca', 'bocca', 'рот', 'فم', 'मुँह', '嘴', '口', '입', 'στόμα', 'פה', 'ปาก', 'ağız', 'mun']),
    ('head'      , ['head', 'Kopf', 'tête', 'cabeza', 'testa', 'голова', 'رأس', 'सिर', '头', '頭', '머리', 'κεφάλι', 'ראש', 'หัว', 'kafa', 'huvud']),
    ('face'      , ['face', 'Gesicht', 'visage', 'cara', 'faccia', 'лицо', 'وجه', 'चेहरा', '脸', '顔', '얼굴', 'πρόσωπο', 'פנים', 'หน้า', 'yüz', 'ansikte']),
    ('hair'      , ['hair', 'Haar', 'cheveux', 'pelo', 'capelli', 'волосы', 'شعر', 'बाल', '头发', '髪', '머리카락', 'μαλλιά', 'שיער', 'ผม', 'saç', 'hår']),
    ('heart'     , ['heart', 'Herz', 'coeur', 'corazón', 'cuore', 'сердце', 'قلب', 'दिल', '心', '心', '심장', 'καρδιά', 'לב', 'หัวใจ', 'kalp', 'hjärta']),
    ('blood'     , ['blood', 'Blut', 'sang', 'sangre', 'sangue', 'кровь', 'دم', 'खून', '血', '血', '피', 'αίμα', 'דם', 'เลือด', 'kan', 'blod']),
    ('bone'      , ['bone', 'Knochen', 'os', 'hueso', 'osso', 'кость', 'عظم', 'हड्डी', '骨', '骨', '뼈', 'κόκαλο', 'עצם', 'กระดูก', 'kemik', 'ben']),
    ('skin'      , ['skin', 'Haut', 'peau', 'piel', 'pelle', 'кожа', 'جلد', 'त्वचा', '皮肤', '皮膚', '피부', 'δέρμα', 'עור', 'ผิว', 'deri', 'hud']),
    ('arm'       , ['arm', 'Arm', 'bras', 'brazo', 'braccio', 'рука', 'ذراع', 'बाँह', '手臂', '腕', '팔', 'βραχίονας', 'זרוع', 'แขน', 'kol', 'arm']),
    ('leg'       , ['leg', 'Bein', 'jambe', 'pierna', 'gamba', 'нога', 'ساق', 'पैर', '腿', '脚', '다리', 'πόδι', 'רגל', 'ขา', 'bacak', 'ben']),
    ('foot'      , ['foot', 'Fuß', 'pied', 'pie', 'piede', 'стопа', 'قدم', 'पैर', '脚', '足', '발', 'πέλμα', 'כף רגל', 'เท้า', 'ayak', 'fot']),
    ('finger'    , ['finger', 'Finger', 'doigt', 'dedo', 'dito', 'палец', 'إصبع', 'उंगली', '手指', '指', '손가락', 'δάκτυλο', 'אצבע', 'นิ้ว', 'parmak', 'finger']),
    ('tooth'     , ['tooth', 'Zahn', 'dent', 'diente', 'dente', 'зуб', 'سن', 'दाँत', '牙', '歯', '이', 'δόντι', 'שן', 'ฟัน', 'diş', 'tand']),
    ('tongue'    , ['tongue', 'Zunge', 'langue', 'lengua', 'lingua', 'язык', 'لسان', 'जीभ', '舌', '舌', '혀', 'γλώσσα', 'לשון', 'ลิ้น', 'dil', 'tunga']),
    ('back'      , ['back', 'Rücken', 'dos', 'espalda', 'schiena', 'спина', 'ظهر', 'पीठ', '背', '背', '등', 'πλάτη', 'גב', 'หลัง', 'sırt', 'rygg']),
    ('neck'      , ['neck', 'Hals', 'cou', 'cuello', 'collo', 'шея', 'رقبة', 'गर्दन', '脖子', '首', '목', 'λαιμός', 'צוואר', 'คอ', 'boyun', 'hals']),
    ('mother'    , ['mother', 'Mutter', 'mère', 'madre', 'madre', 'мать', 'أم', 'माँ', '母', '母', '어머니', 'μητέρα', 'אמא', 'แม่', 'anne', 'moder']),
    ('father'    , ['father', 'Vater', 'père', 'padre', 'padre', 'отец', 'أب', 'पिता', '父', '父', '아버지', 'πατέρας', 'אבא', 'พ่อ', 'baba', 'fader']),
    ('son'       , ['son', 'Sohn', 'fils', 'hijo', 'figlio', 'сын', 'ابن', 'बेटा', '儿子', '息子', '아들', 'γιος', 'בן', 'ลูกชาย', 'oğul', 'son']),
    ('daughter'  , ['daughter', 'Tochter', 'fille', 'hija', 'figlia', 'дочь', 'ابنة', 'बेटी', '女儿', '娘', '딸', 'κόρη', 'בת', 'ลูกสาว', 'kız', 'dotter']),
    ('brother'   , ['brother', 'Bruder', 'frère', 'hermano', 'fratello', 'брат', 'أخ', 'भाई', '兄弟', '兄', '형', 'αδερφός', 'אח', 'พี่ชาย', 'erkek kardeş', 'bror']),
    ('sister'    , ['sister', 'Schwester', 'soeur', 'hermana', 'sorella', 'сестра', 'أخت', 'बहन', '姐妹', '姉', '언니', 'αδερφή', 'אחות', 'พี่สาว', 'kız kardeş', 'syster']),
    ('child'     , ['child', 'Kind', 'enfant', 'niño', 'bambino', 'ребёнок', 'طفل', 'बच्चा', '孩子', '子供', '아이', 'παιδί', 'ילד', 'เด็ก', 'çocuk', 'barn']),
    ('man'       , ['man', 'Mann', 'homme', 'hombre', 'uomo', 'мужчина', 'رجل', 'आदमी', '男人', '男', '남자', 'άντρας', 'איש', 'ผู้ชาย', 'adam', 'man']),
    ('woman'     , ['woman', 'Frau', 'femme', 'mujer', 'donna', 'женщина', 'امرأة', 'औरत', '女人', '女', '여자', 'γυναίκα', 'אישה', 'ผู้หญิง', 'kadın', 'kvinna']),
    ('friend'    , ['friend', 'Freund', 'ami', 'amigo', 'amico', 'друг', 'صديق', 'दोस्त', '朋友', '友達', '친구', 'φίλος', 'חבר', 'เพื่อน', 'arkadaş', 'vän']),
    ('king'      , ['king', 'König', 'roi', 'rey', 're', 'король', 'ملك', 'राजा', '国王', '王', '왕', 'βασιλιάς', 'מלך', 'กษัตริย์', 'kral', 'kung']),
    ('god'       , ['god', 'Gott', 'dieu', 'dios', 'dio', 'бог', 'إله', 'भगवान', '神', '神', '신', 'θεός', 'אל', 'พระเจ้า', 'tanrı', 'gud']),
    ('people'    , ['people', 'Leute', 'gens', 'gente', 'gente', 'люди', 'ناس', 'लोग', '人们', '人々', '사람들', 'άνθρωποι', 'אנשים', 'ผู้คน', 'insanlar', 'folk']),
    ('time'      , ['time', 'Zeit', 'temps', 'tiempo', 'tempo', 'время', 'وقت', 'समय', '时间', '時間', '시간', 'χρόνος', 'זמן', 'เวลา', 'zaman', 'tid']),
    ('year'      , ['year', 'Jahr', 'an', 'año', 'anno', 'год', 'سنة', 'साल', '年', '年', '년', 'χρόνος', 'שנה', 'ปี', 'yıl', 'år']),
    ('month'     , ['month', 'Monat', 'mois', 'mes', 'mese', 'месяц', 'شهر', 'महीना', '月', '月', '월', 'μήνας', 'חודש', 'เดือน', 'ay', 'månad']),
    ('week'      , ['week', 'Woche', 'semaine', 'semana', 'settimana', 'неделя', 'أسبوع', 'हफ्ता', '周', '週', '주', 'εβδομάδα', 'שבוע', 'สัปดาห์', 'hafta', 'vecka']),
    ('hour'      , ['hour', 'Stunde', 'heure', 'hora', 'ora', 'час', 'ساعة', 'घंटा', '小时', '時間', '시간', 'ώρα', 'שעה', 'ชั่วโมง', 'saat', 'timme']),
    ('old'       , ['old', 'alt', 'vieux', 'viejo', 'vecchio', 'старый', 'قديم', 'पुराना', '旧', '古い', '오래된', 'παλιός', 'ישן', 'เก่า', 'eski', 'gammal']),
    ('new'       , ['new', 'neu', 'nouveau', 'nuevo', 'nuovo', 'новый', 'جديد', 'नया', '新', '新しい', '새로운', 'καινούριος', 'חדש', 'ใหม่', 'yeni', 'ny']),
    ('long'      , ['long', 'lang', 'long', 'largo', 'lungo', 'длинный', 'طويل', 'लंबा', '长', '長い', '긴', 'μακρύς', 'ארוך', 'ยาว', 'uzun', 'lång']),
    ('short'     , ['short', 'kurz', 'court', 'corto', 'corto', 'короткий', 'قصير', 'छोटा', '短', '短い', '짧은', 'κοντός', 'קצר', 'สั้น', 'kısa', 'kort']),
    ('run'       , ['run', 'laufen', 'courir', 'correr', 'correre', 'бежать', 'يركض', 'दौड़ना', '跑', '走る', '달리다', 'τρέχω', 'לרוץ', 'วิ่ง', 'koşmak', 'springa']),
    ('walk'      , ['walk', 'gehen', 'marcher', 'caminar', 'camminare', 'идти', 'يمشي', 'चलना', '走', '歩く', '걷다', 'περπατώ', 'ללכת', 'เดิน', 'yürümek', 'gå']),
    ('eat'       , ['eat', 'essen', 'manger', 'comer', 'mangiare', 'есть', 'يأكل', 'खाना', '吃', '食べる', '먹다', 'τρώω', 'לאכול', 'กิน', 'yemek', 'äta']),
    ('drink'     , ['drink', 'trinken', 'boire', 'beber', 'bere', 'пить', 'يشرب', 'पीना', '喝', '飲む', '마시다', 'πίνω', 'לשתות', 'ดื่ม', 'içmek', 'dricka']),
    ('sleep'     , ['sleep', 'schlafen', 'dormir', 'dormir', 'dormire', 'спать', 'ينام', 'सोना', '睡觉', '眠る', '자다', 'κοιμάμαι', 'לישון', 'นอน', 'uyumak', 'sova']),
    ('see'       , ['see', 'sehen', 'voir', 'ver', 'vedere', 'видеть', 'يرى', 'देखना', '看', '見る', '보다', 'βλέπω', 'לראות', 'เห็น', 'görmek', 'se']),
    ('hear'      , ['hear', 'hören', 'entendre', 'oír', 'sentire', 'слышать', 'يسمع', 'सुनना', '听', '聞く', '듣다', 'ακούω', 'לשמוע', 'ได้ยิน', 'duymak', 'höra']),
    ('speak'     , ['speak', 'sprechen', 'parler', 'hablar', 'parlare', 'говорить', 'يتكلم', 'बोलना', '说话', '話す', '말하다', 'μιλώ', 'לדבר', 'พูด', 'konuşmak', 'tala']),
    ('read'      , ['read', 'lesen', 'lire', 'leer', 'leggere', 'читать', 'يقرأ', 'पढ़ना', '读', '読む', '읽다', 'διαβάζω', 'לקרוא', 'อ่าน', 'okumak', 'läsa']),
    ('write'     , ['write', 'schreiben', 'écrire', 'escribir', 'scrivere', 'писать', 'يكتب', 'लिखना', '写', '書く', '쓰다', 'γράφω', 'לכתוב', 'เขียน', 'yazmak', 'skriva']),
    ('think'     , ['think', 'denken', 'penser', 'pensar', 'pensare', 'думать', 'يفكر', 'सोचना', '想', '考える', '생각하다', 'σκέφτομαι', 'לחשוב', 'คิด', 'düşünmek', 'tänka']),
    ('know'      , ['know', 'wissen', 'savoir', 'saber', 'sapere', 'знать', 'يعرف', 'जानना', '知道', '知る', '알다', 'ξέρω', 'לדעת', 'รู้', 'bilmek', 'veta']),
    ('come'      , ['come', 'kommen', 'venir', 'venir', 'venire', 'прийти', 'يأتي', 'आना', '来', '来る', '오다', 'έρχομαι', 'לבוא', 'มา', 'gelmek', 'komma']),
    ('go'        , ['go', 'gehen', 'aller', 'ir', 'andare', 'идти', 'يذهب', 'जाना', '去', '行く', '가다', 'πηγαίνω', 'ללכת', 'ไป', 'gitmek', 'gå']),
    ('give'      , ['give', 'geben', 'donner', 'dar', 'dare', 'давать', 'يعطي', 'देना', '给', 'あげる', '주다', 'δίνω', 'לתת', 'ให้', 'vermek', 'ge']),
    ('take'      , ['take', 'nehmen', 'prendre', 'tomar', 'prendere', 'брать', 'يأخذ', 'लेना', '拿', '取る', '가져가다', 'παίρνω', 'לקחת', 'เอา', 'almak', 'ta']),
    ('make'      , ['make', 'machen', 'faire', 'hacer', 'fare', 'делать', 'يصنع', 'बनाना', '做', '作る', '만들다', 'κάνω', 'לעשות', 'ทำ', 'yapmak', 'göra']),
    ('work'      , ['work', 'Arbeit', 'travail', 'trabajo', 'lavoro', 'работа', 'عمل', 'काम', '工作', '仕事', '일', 'δουλειά', 'עבודה', 'งาน', 'iş', 'arbete']),
    ('love'      , ['love', 'Liebe', 'amour', 'amor', 'amore', 'любовь', 'حب', 'प्यार', '爱', '愛', '사랑', 'αγάπη', 'אהבה', 'ความรัก', 'aşk', 'kärlek']),
    ('die'       , ['die', 'sterben', 'mourir', 'morir', 'morire', 'умирать', 'يموت', 'मरना', '死', '死ぬ', '죽다', 'πεθαίνω', 'למות', 'ตาย', 'ölmek', 'dö']),
    ('live'      , ['live', 'leben', 'vivre', 'vivir', 'vivere', 'жить', 'يعيش', 'जीना', '活', '生きる', '살다', 'ζω', 'לחיות', 'มีชีวิต', 'yaşamak', 'leva']),
    ('fight'     , ['fight', 'kämpfen', 'combattre', 'luchar', 'combattere', 'бороться', 'يقاتل', 'लड़ना', '战斗', '戦う', '싸우다', 'πολεμώ', 'להילחם', 'สู้', 'savaşmak', 'kämpa']),
    ('play'      , ['play', 'spielen', 'jouer', 'jugar', 'giocare', 'играть', 'يلعب', 'खेलना', '玩', '遊ぶ', '놀다', 'παίζω', 'לשחק', 'เล่น', 'oynamak', 'spela']),
    ('learn'     , ['learn', 'lernen', 'apprendre', 'aprender', 'imparare', 'учиться', 'يتعلم', 'सीखना', '学习', '学ぶ', '배우다', 'μαθαίνω', 'ללמוד', 'เรียน', 'öğrenmek', 'lära']),
    ('grow'      , ['grow', 'wachsen', 'grandir', 'crecer', 'crescere', 'расти', 'ينمو', 'बढ़ना', '生长', '育つ', '자라다', 'μεγαλώνω', 'לגדול', 'เติบโต', 'büyümek', 'växa']),
    ('open'      , ['open', 'öffnen', 'ouvrir', 'abrir', 'aprire', 'открыть', 'يفتح', 'खोलना', '打开', '開ける', '열다', 'ανοίγω', 'לפתוח', 'เปิด', 'açmak', 'öppna']),
    ('close'     , ['close', 'schließen', 'fermer', 'cerrar', 'chiudere', 'закрыть', 'يغلق', 'बंद करना', '关闭', '閉める', '닫다', 'κλείνω', 'לסגור', 'ปิด', 'kapatmak', 'stänga']),
    ('find'      , ['find', 'finden', 'trouver', 'encontrar', 'trovare', 'найти', 'يجد', 'ढूंढना', '找', '見つける', '찾다', 'βρίσκω', 'למצוא', 'หา', 'bulmak', 'hitta']),
    ('fall'      , ['fall', 'fallen', 'tomber', 'caer', 'cadere', 'падать', 'يسقط', 'गिरना', '跌落', '落ちる', '넘어지다', 'πέφτω', 'ליפול', 'ล้ม', 'düşmek', 'falla']),
    ('stand'     , ['stand', 'stehen', 'debout', 'estar de pie', 'stare', 'стоять', 'يقف', 'खड़ा होना', '站立', '立つ', '서다', 'στέκομαι', 'לעמוד', 'ยืน', 'durmak', 'stå']),
    ('sit'       , ['sit', 'sitzen', 'asseoir', 'sentarse', 'sedersi', 'сидеть', 'يجلس', 'बैठना', '坐', '座る', '앉다', 'κάθομαι', 'לשבת', 'นั่ง', 'oturmak', 'sitta']),
    ('hold'      , ['hold', 'halten', 'tenir', 'sostener', 'tenere', 'держать', 'يمسك', 'पकड़ना', '握', '持つ', '잡다', 'κρατώ', 'להחזיק', 'ถือ', 'tutmak', 'hålla']),
    ('touch'     , ['touch', 'berühren', 'toucher', 'tocar', 'toccare', 'трогать', 'يلمس', 'छूना', '触摸', '触れる', '만지다', 'αγγίζω', 'לגעת', 'แตะ', 'dokunmak', 'röra']),
    ('build'     , ['build', 'bauen', 'construire', 'construir', 'costruire', 'строить', 'يبني', 'बनाना', '建造', '建てる', '짓다', 'χτίζω', 'לבנות', 'สร้าง', 'inşa etmek', 'bygga']),
    ('cut'       , ['cut', 'schneiden', 'couper', 'cortar', 'tagliare', 'резать', 'يقطع', 'काटना', '切', '切る', '자르다', 'κόβω', 'לחתוך', 'ตัด', 'kesmek', 'skära']),
    ('buy'       , ['buy', 'kaufen', 'acheter', 'comprar', 'comprare', 'купить', 'يشتري', 'खरीदना', '买', '買う', '사다', 'αγοράζω', 'לקנות', 'ซื้อ', 'satın almak', 'köpa']),
    ('sell'      , ['sell', 'verkaufen', 'vendre', 'vender', 'vendere', 'продавать', 'يبيع', 'बेचना', '卖', '売る', '팔다', 'πουλώ', 'למכור', 'ขาย', 'satmak', 'sälja']),
    ('house'     , ['house', 'Haus', 'maison', 'casa', 'casa', 'дом', 'بيت', 'घर', '家', '家', '집', 'σπίτι', 'בית', 'บ้าน', 'ev', 'hus']),
    ('door'      , ['door', 'Tür', 'porte', 'puerta', 'porta', 'дверь', 'باب', 'दरवाजा', '门', '扉', '문', 'πόρτα', 'דלת', 'ประตู', 'kapı', 'dörr']),
    ('window'    , ['window', 'Fenster', 'fenêtre', 'ventana', 'finestra', 'окно', 'نافذة', 'खिड़की', '窗', '窓', '창문', 'παράθυρο', 'חלון', 'หน้าต่าง', 'pencere', 'fönster']),
    ('road'      , ['road', 'Straße', 'route', 'camino', 'strada', 'дорога', 'طريق', 'सड़क', '道路', '道', '도로', 'δρόμος', 'דרך', 'ถนน', 'yol', 'väg']),
    ('bridge'    , ['bridge', 'Brücke', 'pont', 'puente', 'ponte', 'мост', 'جسر', 'पुल', '桥', '橋', '다리', 'γέφυρα', 'גשר', 'สะพาน', 'köprü', 'bro']),
    ('ship'      , ['ship', 'Schiff', 'navire', 'barco', 'nave', 'корабль', 'سفينة', 'जहाज', '船', '船', '배', 'πλοίο', 'ספינה', 'เรือ', 'gemi', 'skepp']),
    ('horse'     , ['horse', 'Pferd', 'cheval', 'caballo', 'cavallo', 'лошадь', 'حصان', 'घोड़ा', '马', '馬', '말', 'άλογο', 'סוס', 'ม้า', 'at', 'häst']),
    ('book'      , ['book', 'Buch', 'livre', 'libro', 'libro', 'книга', 'كتاب', 'किताब', '书', '本', '책', 'βιβλίο', 'ספר', 'หนังสือ', 'kitap', 'bok']),
    ('bread'     , ['bread', 'Brot', 'pain', 'pan', 'pane', 'хлеб', 'خبز', 'रोटी', '面包', 'パン', '빵', 'ψωμί', 'לחם', 'ขนมปัง', 'ekmek', 'bröd']),
    ('food'      , ['food', 'Essen', 'nourriture', 'comida', 'cibo', 'еда', 'طعام', 'खाना', '食物', '食べ物', '음식', 'φαγητό', 'אוכל', 'อาหาร', 'yiyecek', 'mat']),
    ('name'      , ['name', 'Name', 'nom', 'nombre', 'nome', 'имя', 'اسم', 'नाम', '名字', '名前', '이름', 'όνομα', 'שם', 'ชื่อ', 'isim', 'namn']),
    ('word'      , ['word', 'Wort', 'mot', 'palabra', 'parola', 'слово', 'كلمة', 'शब्द', '词', '言葉', '단어', 'λέξη', 'מילה', 'คำ', 'kelime', 'ord']),
    ('world'     , ['world', 'Welt', 'monde', 'mundo', 'mondo', 'мир', 'عالم', 'दुनिया', '世界', '世界', '세계', 'κόσμος', 'עולם', 'โลก', 'dünya', 'värld']),
    ('country'   , ['country', 'Land', 'pays', 'país', 'paese', 'страна', 'بلد', 'देश', '国家', '国', '나라', 'χώρα', 'מדינה', 'ประเทศ', 'ülke', 'land']),
    ('city'      , ['city', 'Stadt', 'ville', 'ciudad', 'città', 'город', 'مدينة', 'शहर', '城市', '都市', '도시', 'πόλη', 'עיר', 'เมือง', 'şehir', 'stad']),
    ('war'       , ['war', 'Krieg', 'guerre', 'guerra', 'guerra', 'война', 'حرب', 'युद्ध', '战争', '戦争', '전쟁', 'πόλεμος', 'מלחמה', 'สงคราม', 'savaş', 'krig']),
    ('peace'     , ['peace', 'Frieden', 'paix', 'paz', 'pace', 'мир', 'سلام', 'शांति', '和平', '平和', '평화', 'ειρήνη', 'שלום', 'สันติภาพ', 'barış', 'fred']),
    ('power'     , ['power', 'Macht', 'pouvoir', 'poder', 'potere', 'власть', 'قوة', 'शक्ति', '力量', '力', '힘', 'δύναμη', 'כוח', 'พลัง', 'güç', 'makt']),
    ('money'     , ['money', 'Geld', 'argent', 'dinero', 'soldi', 'деньги', 'مال', 'पैसा', '钱', 'お金', '돈', 'χρήματα', 'כסף', 'เงิน', 'para', 'pengar']),
    ('gold'      , ['gold', 'Gold', 'or', 'oro', 'oro', 'золото', 'ذهب', 'सोना', '金', '金', '금', 'χρυσός', 'זהב', 'ทอง', 'altın', 'guld']),
    ('iron'      , ['iron', 'Eisen', 'fer', 'hierro', 'ferro', 'железо', 'حديد', 'लोहा', '铁', '鉄', '철', 'σίδηρος', 'ברזל', 'เหล็ก', 'demir', 'järn']),
    ('sword'     , ['sword', 'Schwert', 'épée', 'espada', 'spada', 'меч', 'سيف', 'तलवार', '剑', '剣', '검', 'σπαθί', 'חרב', 'ดาบ', 'kılıç', 'svärd']),
    ('red'       , ['red', 'rot', 'rouge', 'rojo', 'rosso', 'красный', 'أحمر', 'लाल', '红', '赤', '빨간', 'κόκκινο', 'אדום', 'แดง', 'kırmızı', 'röd']),
    ('white'     , ['white', 'weiß', 'blanc', 'blanco', 'bianco', 'белый', 'أبيض', 'सफेद', '白', '白', '흰', 'λευκό', 'לבן', 'ขาว', 'beyaz', 'vit']),
    ('black'     , ['black', 'schwarz', 'noir', 'negro', 'nero', 'чёрный', 'أسود', 'काला', '黑', '黒', '검은', 'μαύρο', 'שחור', 'ดำ', 'siyah', 'svart']),
    ('green'     , ['green', 'grün', 'vert', 'verde', 'verde', 'зелёный', 'أخضر', 'हरा', '绿', '緑', '녹색', 'πράσινο', 'ירוק', 'เขียว', 'yeşil', 'grön']),
    ('blue'      , ['blue', 'blau', 'bleu', 'azul', 'blu', 'синий', 'أزرق', 'नीला', '蓝', '青', '파란', 'μπλε', 'כחול', 'น้ำเงิน', 'mavi', 'blå']),
    ('yellow'    , ['yellow', 'gelb', 'jaune', 'amarillo', 'giallo', 'жёлтый', 'أصفر', 'पीला', '黄', '黄', '노란', 'κίτρινο', 'צהוב', 'เหลือง', 'sarı', 'gul']),
    ('one'       , ['one', 'eins', 'un', 'uno', 'uno', 'один', 'واحد', 'एक', '一', '一', '일', 'ένα', 'אחד', 'หนึ่ง', 'bir', 'en']),
    ('two'       , ['two', 'zwei', 'deux', 'dos', 'due', 'два', 'اثنان', 'दो', '二', '二', '이', 'δύο', 'שניים', 'สอง', 'iki', 'två']),
    ('three'     , ['three', 'drei', 'trois', 'tres', 'tre', 'три', 'ثلاثة', 'तीन', '三', '三', '삼', 'τρία', 'שלוש', 'สาม', 'üç', 'tre']),
    ('ten'       , ['ten', 'zehn', 'dix', 'diez', 'dieci', 'десять', 'عشرة', 'दस', '十', '十', '십', 'δέκα', 'עשר', 'สิบ', 'on', 'tio']),
    ('hundred'   , ['hundred', 'hundert', 'cent', 'ciento', 'cento', 'сто', 'مئة', 'सौ', '百', '百', '백', 'εκατό', 'מאה', 'ร้อย', 'yüz', 'hundra']),
    ('thousand'  , ['thousand', 'tausend', 'mille', 'mil', 'mille', 'тысяча', 'ألف', 'हज़ार', '千', '千', '천', 'χίλια', 'אלף', 'พัน', 'bin', 'tusen']),
    ('life'      , ['life', 'Leben', 'vie', 'vida', 'vita', 'жизнь', 'حياة', 'जीवन', '生命', '生命', '생명', 'ζωή', 'חיים', 'ชีวิต', 'hayat', 'liv']),
    ('death'     , ['death', 'Tod', 'mort', 'muerte', 'morte', 'смерть', 'موت', 'मृत्यु', '死亡', '死', '죽음', 'θάνατος', 'מוות', 'ความตาย', 'ölüm', 'död']),
    ('truth'     , ['truth', 'Wahrheit', 'vérité', 'verdad', 'verità', 'правда', 'حقيقة', 'सच', '真相', '真実', '진실', 'αλήθεια', 'אמת', 'ความจริง', 'gerçek', 'sanning']),
    ('dream'     , ['dream', 'Traum', 'rêve', 'sueño', 'sogno', 'мечта', 'حلم', 'सपना', '梦', '夢', '꿈', 'όνειρο', 'חלום', 'ความฝัน', 'rüya', 'dröm']),
    ('fear'      , ['fear', 'Angst', 'peur', 'miedo', 'paura', 'страх', 'خوف', 'डर', '恐惧', '恐怖', '두려움', 'φόβος', 'פחד', 'ความกลัว', 'korku', 'rädsla']),
    ('hope'      , ['hope', 'Hoffnung', 'espoir', 'esperanza', 'speranza', 'надежда', 'أمل', 'उम्मीद', '希望', '希望', '희망', 'ελπίδα', 'תקווה', 'ความหวัง', 'umut', 'hopp']),
    ('beauty'    , ['beauty', 'Schönheit', 'beauté', 'belleza', 'bellezza', 'красота', 'جمال', 'सुंदरता', '美', '美', '아름다움', 'ομορφιά', 'יופי', 'ความงาม', 'güzellik', 'skönhet']),
    ('freedom'   , ['freedom', 'Freiheit', 'liberté', 'libertad', 'libertà', 'свобода', 'حرية', 'आज़ादी', '自由', '自由', '자유', 'ελευθερία', 'חופש', 'อิสรภาพ', 'özgürlük', 'frihet']),
    ('justice'   , ['justice', 'Gerechtigkeit', 'justice', 'justicia', 'giustizia', 'справедливость', 'عدالة', 'न्याय', '正义', '正義', '정의', 'δικαιοσύνη', 'צדק', 'ความยุติธรรม', 'adalet', 'rättvisa']),
    ('wisdom'    , ['wisdom', 'Weisheit', 'sagesse', 'sabiduría', 'saggezza', 'мудрость', 'حكمة', 'ज्ञान', '智慧', '知恵', '지혜', 'σοφία', 'חוכמה', 'ปัญญา', 'bilgelik', 'visdom']),
    ('dog'       , ['dog', 'Hund', 'chien', 'perro', 'cane', 'собака', 'كلب', 'कुत्ता', '狗', '犬', '개', 'σκύλος', 'כלב', 'หมา', 'köpek', 'hund']),
    ('cat'       , ['cat', 'Katze', 'chat', 'gato', 'gatto', 'кошка', 'قطة', 'बिल्ली', '猫', '猫', '고양이', 'γάτα', 'חתול', 'แมว', 'kedi', 'katt']),
    ('bird'      , ['bird', 'Vogel', 'oiseau', 'pájaro', 'uccello', 'птица', 'طائر', 'पक्षी', '鸟', '鳥', '새', 'πουλί', 'ציפור', 'นก', 'kuş', 'fågel']),
    ('fish'      , ['fish', 'Fisch', 'poisson', 'pez', 'pesce', 'рыба', 'سمكة', 'मछली', '鱼', '魚', '물고기', 'ψάρι', 'דג', 'ปลา', 'balık', 'fisk']),
    ('wolf'      , ['wolf', 'Wolf', 'loup', 'lobo', 'lupo', 'волк', 'ذئب', 'भेड़िया', '狼', 'オオカミ', '늑대', 'λύκος', 'זאב', 'หมาป่า', 'kurt', 'varg']),
    ('snake'     , ['snake', 'Schlange', 'serpent', 'serpiente', 'serpente', 'змея', 'ثعبان', 'साँप', '蛇', '蛇', '뱀', 'φίδι', 'נחש', 'งู', 'yılan', 'orm']),
    ('eagle'     , ['eagle', 'Adler', 'aigle', 'águila', 'aquila', 'орёл', 'نسر', 'चील', '鹰', '鷹', '독수리', 'αετός', 'נשר', 'นกอินทรี', 'kartal', 'örn']),
    ('milk'      , ['milk', 'Milch', 'lait', 'leche', 'latte', 'молоко', 'حليب', 'दूध', '牛奶', '牛乳', '우유', 'γάλα', 'חלב', 'นม', 'süt', 'mjölk']),
    ('wine'      , ['wine', 'Wein', 'vin', 'vino', 'vino', 'вино', 'نبيذ', 'शराब', '葡萄酒', 'ワイン', '포도주', 'κρασί', 'יין', 'ไวน์', 'şarap', 'vin']),
    ('salt'      , ['salt', 'Salz', 'sel', 'sal', 'sale', 'соль', 'ملح', 'नमक', '盐', '塩', '소금', 'αλάτι', 'מלח', 'เกลือ', 'tuz', 'salt']),
    ('fruit'     , ['fruit', 'Frucht', 'fruit', 'fruta', 'frutto', 'фрукт', 'فاكهة', 'फल', '水果', '果物', '과일', 'φρούτο', 'פרי', 'ผลไม้', 'meyve', 'frukt']),
    ('meat'      , ['meat', 'Fleisch', 'viande', 'carne', 'carne', 'мясо', 'لحم', 'मांस', '肉', '肉', '고기', 'κρέας', 'בשר', 'เนื้อ', 'et', 'kött']),
    ('island'    , ['island', 'Insel', 'île', 'isla', 'isola', 'остров', 'جزيرة', 'द्वीप', '岛', '島', '섬', 'νησί', 'אי', 'เกาะ', 'ada', 'ö']),
    ('cave'      , ['cave', 'Höhle', 'grotte', 'cueva', 'grotta', 'пещера', 'كهف', 'गुफा', '洞穴', '洞窟', '동굴', 'σπηλιά', 'מערה', 'ถ้ำ', 'mağara', 'grotta']),
    ('field'     , ['field', 'Feld', 'champ', 'campo', 'campo', 'поле', 'حقل', 'मैदान', '田野', '畑', '들판', 'χωράφι', 'שדה', 'ทุ่ง', 'tarla', 'fält']),
    ('desert'    , ['desert', 'Wüste', 'désert', 'desierto', 'deserto', 'пустыня', 'صحراء', 'रेगिस्तान', '沙漠', '砂漠', '사막', 'έρημος', 'מדבר', 'ทะเลทราย', 'çöl', 'öken']),
    ('valley'    , ['valley', 'Tal', 'vallée', 'valle', 'valle', 'долина', 'وادي', 'घाटी', '山谷', '谷', '계곡', 'κοιλάδα', 'עמק', 'หุบเขา', 'vadi', 'dal']),
    ('language'  , ['language', 'Sprache', 'langue', 'idioma', 'lingua', 'язык', 'لغة', 'भाषा', '语言', '言語', '언어', 'γλώσσα', 'שפה', 'ภาษา', 'dil', 'språk']),
    ('music'     , ['music', 'Musik', 'musique', 'música', 'musica', 'музыка', 'موسيقى', 'संगीत', '音乐', '音楽', '음악', 'μουσική', 'מוזיקה', 'ดนตรี', 'müzik', 'musik']),
    ('art'       , ['art', 'Kunst', 'art', 'arte', 'arte', 'искусство', 'فن', 'कला', '艺术', '芸術', '예술', 'τέχνη', 'אמנות', 'ศิลปะ', 'sanat', 'konst']),
    ('science'   , ['science', 'Wissenschaft', 'science', 'ciencia', 'scienza', 'наука', 'علم', 'विज्ञान', '科学', '科学', '과학', 'επιστήμη', 'מדע', 'วิทยาศาสตร์', 'bilim', 'vetenskap']),
    ('history'   , ['history', 'Geschichte', 'histoire', 'historia', 'storia', 'история', 'تاريخ', 'इतिहास', '历史', '歴史', '역사', 'ιστορία', 'היסטוריה', 'ประวัติศาสตร์', 'tarih', 'historia']),
    ('school'    , ['school', 'Schule', 'école', 'escuela', 'scuola', 'школа', 'مدرسة', 'स्कूल', '学校', '学校', '학교', 'σχολείο', 'בית ספר', 'โรงเรียน', 'okul', 'skola']),
    ('wall'      , ['wall', 'Wand', 'mur', 'pared', 'muro', 'стена', 'جدار', 'दीवार', '墙', '壁', '벽', 'τοίχος', 'קיר', 'กำแพง', 'duvar', 'vägg']),
    ('floor'     , ['floor', 'Boden', 'sol', 'suelo', 'pavimento', 'пол', 'أرضية', 'फर्श', '地板', '床', '바닥', 'πάτωμα', 'רצפה', 'พื้น', 'zemin', 'golv']),
    ('table'     , ['table', 'Tisch', 'table', 'mesa', 'tavolo', 'стол', 'طاولة', 'मेज', '桌子', 'テーブル', '탁자', 'τραπέζι', 'שולחן', 'โต๊ะ', 'masa', 'bord']),
    ('chair'     , ['chair', 'Stuhl', 'chaise', 'silla', 'sedia', 'стул', 'كرسي', 'कुर्सी', '椅子', '椅子', '의자', 'καρέκλα', 'כיסא', 'เก้าอี้', 'sandalye', 'stol']),
    ('bed'       , ['bed', 'Bett', 'lit', 'cama', 'letto', 'кровать', 'سرير', 'बिस्तर', '床', 'ベッド', '침대', 'κρεβάτι', 'מיטה', 'เตียง', 'yatak', 'säng']),
    ('knife'     , ['knife', 'Messer', 'couteau', 'cuchillo', 'coltello', 'нож', 'سكين', 'चाकू', '刀', 'ナイフ', '칼', 'μαχαίρι', 'סכין', 'มีด', 'bıçak', 'kniv']),
    ('key'       , ['key', 'Schlüssel', 'clé', 'llave', 'chiave', 'ключ', 'مفتاح', 'चाबी', '钥匙', '鍵', '열쇠', 'κλειδί', 'מפתח', 'กุญแจ', 'anahtar', 'nyckel']),
    ('wheel'     , ['wheel', 'Rad', 'roue', 'rueda', 'ruota', 'колесо', 'عجلة', 'पहिया', '轮子', '車輪', '바퀴', 'ρόδα', 'גלגל', 'ล้อ', 'tekerlek', 'hjul']),
    ('cup'       , ['cup', 'Tasse', 'tasse', 'taza', 'tazza', 'чашка', 'كوب', 'प्याला', '杯子', 'カップ', '컵', 'κούπα', 'כוס', 'ถ้วย', 'bardak', 'kopp']),
    ('big'       , ['big', 'groß', 'grand', 'grande', 'grande', 'большой', 'كبير', 'बड़ा', '大', '大きい', '큰', 'μεγάλος', 'גדול', 'ใหญ่', 'büyük', 'stor']),
    ('small'     , ['small', 'klein', 'petit', 'pequeño', 'piccolo', 'маленький', 'صغير', 'छوटا', '小', '小さい', '작은', 'μικρός', 'קטן', 'เล็ก', 'küçük', 'liten']),
    ('hot'       , ['hot', 'heiß', 'chaud', 'caliente', 'caldo', 'горячий', 'حار', 'गरम', '热', '熱い', '뜨거운', 'ζεστός', 'חם', 'ร้อน', 'sıcak', 'het']),
    ('cold'      , ['cold', 'kalt', 'froid', 'frío', 'freddo', 'холодный', 'بارد', 'ठंडा', '冷', '冷たい', '차가운', 'κρύος', 'קר', 'เย็น', 'soğuk', 'kall']),
    ('good'      , ['good', 'gut', 'bon', 'bueno', 'buono', 'хороший', 'جيد', 'अच्छा', '好', '良い', '좋은', 'καλός', 'טוב', 'ดี', 'iyi', 'bra']),
    ('bad'       , ['bad', 'schlecht', 'mauvais', 'malo', 'cattivo', 'плохой', 'سيئ', 'बुरा', '坏', '悪い', '나쁜', 'κακός', 'רע', 'เลว', 'kötü', 'dålig']),
    ('high'      , ['high', 'hoch', 'haut', 'alto', 'alto', 'высокий', 'عالٍ', 'ऊँचा', '高', '高い', '높은', 'ψηλός', 'גבוה', 'สูง', 'yüksek', 'hög']),
    ('deep'      , ['deep', 'tief', 'profond', 'profundo', 'profondo', 'глубокий', 'عميق', 'गहरा', '深', '深い', '깊은', 'βαθύς', 'עמוק', 'ลึก', 'derin', 'djup']),
    ('rope'      , ['rope', 'Seil', 'corde', 'cuerda', 'corda', 'верёвка', 'حبل', 'रस्सी', '绳子', 'ロープ', '밧줄', 'σχοινί', 'חבל', 'เชือก', 'ip', 'rep']),
    ('mirror'    , ['mirror', 'Spiegel', 'miroir', 'espejo', 'specchio', 'зеркало', 'مرآة', 'आईना', '镜子', '鏡', '거울', 'καθρέφτης', 'מראה', 'กระจก', 'ayna', 'spegel']),
    ('fast'      , ['fast', 'schnell', 'rapide', 'rápido', 'veloce', 'быстрый', 'سريع', 'तेज', '快', '速い', '빠른', 'γρήγορος', 'מהיר', 'เร็ว', 'hızlı', 'snabb']),
    ('slow'      , ['slow', 'langsam', 'lent', 'lento', 'lento', 'медленный', 'بطيء', 'धीमा', '慢', '遅い', '느린', 'αργός', 'איטי', 'ช้า', 'yavaş', 'långsam']),
    ('hard'      , ['hard', 'hart', 'dur', 'duro', 'duro', 'твёрдый', 'صعب', 'कठिन', '硬', '硬い', '딱딱한', 'σκληρός', 'קשה', 'แข็ง', 'sert', 'hård']),
    ('soft'      , ['soft', 'weich', 'doux', 'suave', 'morbido', 'мягкий', 'ناعم', 'नरम', '软', '柔らかい', '부드러운', 'μαλακός', 'רך', 'นุ่ม', 'yumuşak', 'mjuk']),
    ('full'      , ['full', 'voll', 'plein', 'lleno', 'pieno', 'полный', 'ممتلئ', 'भरा', '满', 'いっぱい', '가득한', 'γεμάτος', 'מלא', 'เต็ม', 'dolu', 'full']),
    ('empty'     , ['empty', 'leer', 'vide', 'vacío', 'vuoto', 'пустой', 'فارغ', 'खाली', '空', '空', '빈', 'άδειος', 'ריק', 'ว่าง', 'boş', 'tom']),
]
SEMANTIC_CONCEPTS += [
    ('government'  , ['government', 'Regierung', 'gouvernement', 'gobierno', 'governo', 'правительство', 'حكومة', 'सरकार', '政府', '政府', '정부', 'κυβέρνηση', 'ממשלה', 'รัฐบาล', 'hükümet', 'regering']),
    ('president'   , ['president', 'Präsident', 'président', 'presidente', 'presidente', 'президент', 'رئيس', 'राष्ट्रपति', '总统', '大統領', '대통령', 'πρόεδρος', 'נשיא', 'ประธาน', 'cumhurbaşkanı', 'president']),
    ('election'    , ['election', 'Wahl', 'élection', 'elección', 'elezione', 'выборы', 'انتخابات', 'चुनाव', '选举', '選挙', '선거', 'εκλογές', 'בחירות', 'การเลือกตั้ง', 'seçim', 'val']),
    ('parliament'  , ['parliament', 'Parlament', 'parlement', 'parlamento', 'parlamento', 'парламент', 'برلمان', 'संसद', '议会', '議会', '의회', 'κοινοβούλιο', 'פרלמנט', 'รัฐสภา', 'parlamento', 'parlament']),
    ('army'        , ['army', 'Armee', 'armée', 'ejército', 'esercito', 'армия', 'جيش', 'सेना', '军队', '軍隊', '군대', 'στρατός', 'צבא', 'กองทัพ', 'ordu', 'armé']),
    ('police'      , ['police', 'Polizei', 'police', 'policía', 'polizia', 'полиция', 'شرطة', 'पुलिस', '警察', '警察', '경찰', 'αστυνομία', 'משטרה', 'ตำรวจ', 'polis', 'polis']),
    ('law'         , ['law', 'Gesetz', 'loi', 'ley', 'legge', 'закон', 'قانون', 'कानून', '法律', '法律', '법률', 'νόμος', 'חוק', 'กฎหมาย', 'kanun', 'lag']),
    ('party'       , ['party', 'Partei', 'parti', 'partido', 'partito', 'партия', 'حزب', 'पार्टी', '政党', '政党', '정당', 'κόμμα', 'מפלגה', 'พรรค', 'parti', 'parti']),
    ('minister'    , ['minister', 'Minister', 'ministre', 'ministro', 'ministro', 'министр', 'وزير', 'मंत्री', '部长', '大臣', '장관', 'υπουργός', 'שר', 'รัฐมนตรี', 'bakan', 'minister']),
    ('capital'     , ['capital', 'Hauptstadt', 'capitale', 'capital', 'capitale', 'столица', 'عاصمة', 'राजधानी', '首都', '首都', '수도', 'πρωτεύουσα', 'בירה', 'เมืองหลวง', 'başkent', 'huvudstad']),
    ('region'      , ['region', 'Region', 'région', 'región', 'regione', 'регион', 'منطقة', 'क्षेत्र', '地区', '地域', '지역', 'περιοχή', 'אזור', 'ภูมิภาค', 'bölge', 'region']),
    ('border'      , ['border', 'Grenze', 'frontière', 'frontera', 'confine', 'граница', 'حدود', 'सीमा', '边境', '国境', '국경', 'σύνορα', 'גבול', 'ชายแดน', 'sınır', 'gräns']),
    ('population'  , ['population', 'Bevölkerung', 'population', 'población', 'popolazione', 'население', 'سكان', 'जनसंख्या', '人口', '人口', '인구', 'πληθυσμός', 'אוכלוסייה', 'ประชากร', 'nüfus', 'befolkning']),
    ('north'       , ['north', 'Norden', 'nord', 'norte', 'nord', 'север', 'شمال', 'उत्तर', '北', '北', '북', 'βορράς', 'צפון', 'เหนือ', 'kuzey', 'norr']),
    ('south'       , ['south', 'Süden', 'sud', 'sur', 'sud', 'юг', 'جنوب', 'दक्षिण', '南', '南', '남', 'νότος', 'דרום', 'ใต้', 'güney', 'söder']),
    ('east'        , ['east', 'Osten', 'est', 'este', 'est', 'восток', 'شرق', 'पूर्व', '东', '東', '동', 'ανατολή', 'מזרח', 'ตะวันออก', 'doğu', 'öst']),
    ('west'        , ['west', 'Westen', 'ouest', 'oeste', 'ovest', 'запад', 'غرب', 'पश्चिम', '西', '西', '서', 'δύση', 'מערב', 'ตะวันตก', 'batı', 'väst']),
    ('religion'    , ['religion', 'Religion', 'religion', 'religión', 'religione', 'религия', 'دين', 'धर्म', '宗教', '宗教', '종교', 'θρησκεία', 'דת', 'ศาสนา', 'din', 'religion']),
    ('prayer'      , ['prayer', 'Gebet', 'prière', 'oración', 'preghiera', 'молитва', 'صلاة', 'प्रार्थना', '祈祷', '祈り', '기도', 'προσευχή', 'תפילה', 'การอธิษฐาน', 'dua', 'bön']),
    ('church'      , ['church', 'Kirche', 'église', 'iglesia', 'chiesa', 'церковь', 'كنيسة', 'चर्च', '教堂', '教会', '교회', 'εκκλησία', 'כנסייה', 'โบสถ์', 'kilise', 'kyrka']),
    ('mosque'      , ['mosque', 'Moschee', 'mosquée', 'mezquita', 'moschea', 'мечеть', 'مسجد', 'मस्जिद', '清真寺', 'モスク', '모스크', 'τζαμί', 'מסגד', 'มัสยิด', 'cami', 'moské']),
    ('temple'      , ['temple', 'Tempel', 'temple', 'templo', 'tempio', 'храм', 'معبد', 'मंदिर', '寺庙', '寺院', '사원', 'ναός', 'מקדש', 'วัด', 'tapınak', 'tempel']),
    ('prophet'     , ['prophet', 'Prophet', 'prophète', 'profeta', 'profeta', 'пророк', 'نبي', 'नबी', '先知', '預言者', '예언자', 'προφήτης', 'נביא', 'ผู้เผยพระวจนะ', 'peygamber', 'profet']),
    ('economy'     , ['economy', 'Wirtschaft', 'économie', 'economía', 'economia', 'экономика', 'اقتصاد', 'अर्थव्यवस्था', '经济', '経済', '경제', 'οικονομία', 'כלכלה', 'เศรษฐกิจ', 'ekonomi', 'ekonomi']),
    ('bank'        , ['bank', 'Bank', 'banque', 'banco', 'banca', 'банк', 'بنك', 'बैंक', '银行', '銀行', '은행', 'τράπεζα', 'בנק', 'ธนาคาร', 'banka', 'bank']),
    ('market'      , ['market', 'Markt', 'marché', 'mercado', 'mercato', 'рынок', 'سوق', 'बाज़ार', '市场', '市場', '시장', 'αγορά', 'שוק', 'ตลาด', 'pazar', 'marknad']),
    ('trade'       , ['trade', 'Handel', 'commerce', 'comercio', 'commercio', 'торговля', 'تجارة', 'व्यापार', '贸易', '貿易', '무역', 'εμπόριο', 'סחר', 'การค้า', 'ticaret', 'handel']),
    ('oil'         , ['oil', 'Öl', 'pétrole', 'petróleo', 'petrolio', 'нефть', 'نفط', 'तेल', '石油', '石油', '석유', 'πετρέλαιο', 'נפט', 'น้ำมัน', 'petrol', 'olja']),
    ('university'  , ['university', 'Universität', 'université', 'universidad', 'università', 'университет', 'جامعة', 'विश्वविद्यालय', '大学', '大学', '대학교', 'πανεπιστήμιο', 'אוניברסיטה', 'มหาวิทยาลัย', 'üniversite', 'universitet']),
    ('hospital'    , ['hospital', 'Krankenhaus', 'hôpital', 'hospital', 'ospedale', 'больница', 'مستشفى', 'अस्पताल', '医院', '病院', '병원', 'νοσοκομείο', 'בית חולים', 'โรงพยาบาล', 'hastane', 'sjukhus']),
    ('newspaper'   , ['newspaper', 'Zeitung', 'journal', 'periódico', 'giornale', 'газета', 'صحيفة', 'अखबार', '报纸', '新聞', '신문', 'εφημερίδα', 'עיתון', 'หนังสือพิมพ์', 'gazete', 'tidning']),
    ('television'  , ['television', 'Fernsehen', 'télévision', 'televisión', 'televisione', 'телевидение', 'تلفزيون', 'टेलीविजन', '电视', 'テレビ', '텔레비전', 'τηλεόραση', 'טלוויזיה', 'โทรทัศน์', 'televizyon', 'television']),
    ('internet'    , ['internet', 'Internet', 'internet', 'internet', 'internet', 'интернет', 'إنترنت', 'इंटरनेट', '互联网', 'インターネット', '인터넷', 'διαδίκτυο', 'אינטרנט', 'อินเทอร์เน็ต', 'internet', 'internet']),
    ('sport'       , ['sport', 'Sport', 'sport', 'deporte', 'sport', 'спорт', 'رياضة', 'खेल', '体育', 'スポーツ', '스포츠', 'άθλημα', 'ספורט', 'กีฬา', 'spor', 'sport']),
    ('football'    , ['football', 'Fußball', 'football', 'fútbol', 'calcio', 'футбол', 'كرة القدم', 'फुटबॉल', '足球', 'サッカー', '축구', 'ποδόσφαιρο', 'כדורגל', 'ฟุตบอล', 'futbol', 'fotboll']),
    ('film'        , ['film', 'Film', 'film', 'película', 'film', 'фильм', 'فيلم', 'फिल्म', '电影', '映画', '영화', 'ταινία', 'סרט', 'ภาพยนตร์', 'film', 'film']),
    ('century'     , ['century', 'Jahrhundert', 'siècle', 'siglo', 'secolo', 'век', 'قرن', 'सदी', '世纪', '世紀', '세기', 'αιώνας', 'מאה', 'ศตวรรษ', 'yüzyıl', 'århundrade']),
    ('war'         , ['war', 'Krieg', 'guerre', 'guerra', 'guerra', 'война', 'حرب', 'युद्ध', '战争', '戦争', '전쟁', 'πόλεμος', 'מלחמה', 'สงคราม', 'savaş', 'krig']),
    ('revolution'  , ['revolution', 'Revolution', 'révolution', 'revolución', 'rivoluzione', 'революция', 'ثورة', 'क्रांति', '革命', '革命', '혁명', 'επανάσταση', 'מהפכה', 'การปฏิวัติ', 'devrim', 'revolution']),
    ('empire'      , ['empire', 'Reich', 'empire', 'imperio', 'impero', 'империя', 'إمبراطورية', 'साम्राज्य', '帝国', '帝国', '제국', 'αυτοκρατορία', 'אימפריה', 'จักรวรรดิ', 'imparatorluk', 'imperium']),
    ('planet'      , ['planet', 'Planet', 'planète', 'planeta', 'pianeta', 'планета', 'كوكب', 'ग्रह', '行星', '惑星', '행성', 'πλανήτης', 'כוכב לכת', 'ดาวเคราะห์', 'gezegen', 'planet']),
    ('animal'      , ['animal', 'Tier', 'animal', 'animal', 'animale', 'животное', 'حيوان', 'जानवर', '动物', '動物', '동물', 'ζώο', 'בעל חיים', 'สัตว์', 'hayvan', 'djur']),
    ('plant'       , ['plant', 'Pflanze', 'plante', 'planta', 'pianta', 'растение', 'نبات', 'पौधा', '植物', '植物', '식물', 'φυτό', 'צמח', 'พืช', 'bitki', 'växt']),
    ('energy'      , ['energy', 'Energie', 'énergie', 'energía', 'energia', 'энергия', 'طاقة', 'ऊर्जा', '能量', 'エネルギー', '에너지', 'ενέργεια', 'אנרגיה', 'พลังงาน', 'enerji', 'energi']),
    ('machine'     , ['machine', 'Maschine', 'machine', 'máquina', 'macchina', 'машина', 'آلة', 'मशीन', '机器', '機械', '기계', 'μηχανή', 'מכונה', 'เครื่องจักร', 'makine', 'maskin']),
    ('computer'    , ['computer', 'Computer', 'ordinateur', 'computadora', 'computer', 'компьютер', 'حاسوب', 'कंप्यूटर', '电脑', 'コンピュータ', '컴퓨터', 'υπολογιστής', 'מחשב', 'คอมพิวเตอร์', 'bilgisayar', 'dator']),
    ('telephone'   , ['telephone', 'Telefon', 'téléphone', 'teléfono', 'telefono', 'телефон', 'هاتف', 'फोन', '电话', '電話', '전화', 'τηλέφωνο', 'טלפון', 'โทรศัพท์', 'telefon', 'telefon']),
    ('medicine'    , ['medicine', 'Medizin', 'médecine', 'medicina', 'medicina', 'медицина', 'طب', 'चिकित्सा', '医学', '医学', '의학', 'ιατρική', 'רפואה', 'การแพทย์', 'tıp', 'medicin']),
    ('disease'     , ['disease', 'Krankheit', 'maladie', 'enfermedad', 'malattia', 'болезнь', 'مرض', 'बीमारी', '疾病', '病気', '질병', 'ασθένεια', 'מחלה', 'โรค', 'hastalık', 'sjukdom']),
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
    def __init__(self, n_pairs=100000, vocab_dir="vocabularies"):
        self.pairs = []

        # Load vocab
        wiki_vocab = load_wiki_vocab(vocab_dir)
        if wiki_vocab:
            word_list = [w for words in wiki_vocab.values() for w in words]
            print(f"  Wikipedia vocab: {len(wiki_vocab)} languages, {len(word_list):,} words")
        else:
            word_list = [w for ws in MULTILINGUAL_WORDS.values() for w in ws]
            print(f"  Built-in vocab: {len(word_list)} words")

        n_font = n_pairs // 2
        n_sem  = n_pairs - n_font
        raw_pairs = []

        for _ in range(n_font):
            word = random.choice(word_list)
            sz_a = random.choice([17, 18, 19, 20, 21, 22, 23])
            sz_b = random.choice([17, 18, 19, 20, 21, 22, 23])
            raw_pairs.append((word, sz_a, word, sz_b))

        for _ in range(n_sem):
            _, translations = random.choice(SEMANTIC_CONCEPTS)
            a = random.choice(translations)
            b = random.choice(translations)
            while b == a:
                b = random.choice(translations)
            raw_pairs.append((a, FONT_SIZE, b, FONT_SIZE))

        random.shuffle(raw_pairs)

        print(f"  Pre-rendering {len(raw_pairs):,} pairs into RAM...")
        for wa, sa, wb, sb in tqdm(raw_pairs, desc="  rendering"):
            self.pairs.append((word_to_tensor(wa, sa), word_to_tensor(wb, sb)))
        print(f"  Pre-rendering done.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

# ─── TRAINING ────────────────────────────────────────────────────────────────

def train(model, epochs=EPOCHS, n_pairs=100000, lr=LR):
    print(f"\n=== TRAINING on {DEVICE} ===")
    print(f"  Pairs: {n_pairs}  Batch: {BATCH_SIZE}  Epochs: {epochs}")

    dataset = WordPairDataset(n_pairs=n_pairs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=True)

    criterion = NTXentLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for ta, tb in loader:
            ta = ta.to(DEVICE)
            tb = tb.to(DEVICE)

            optimizer.zero_grad()
            emb_a = model(ta)
            emb_b = model(tb)
            loss = criterion(emb_a, emb_b)
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

    # Build model — resume from checkpoint if available
    print("\n=== MODEL ===")
    if os.path.exists("visual_embeddings.pt"):
        model = load_model("visual_embeddings.pt")
        print(f"  Resumed from visual_embeddings.pt")
    else:
        model = VisualWordEncoder(embed_dim=EMBED_DIM).to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
        print(f"  Embed dim:  {EMBED_DIM}")
        validate_similarity(model, "before training")

    # Train
    model = train(model, epochs=EPOCHS, n_pairs=100000, lr=LR)
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
