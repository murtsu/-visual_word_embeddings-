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

IMG_W       = 384
IMG_H       = 96
EMBED_DIM   = 256
BATCH_SIZE  = 128
LR          = 3e-4
EPOCHS      = 100
TEMPERATURE = 0.07
FONT_SIZE   = 60
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
        Conv(1→32, k=3) → GN → ReLU → MaxPool(2)
        Conv(32→64, k=3) → GN → ReLU → MaxPool(2)
        Conv(64→128, k=3) → GN → ReLU → MaxPool(2)
        Conv(128→256, k=3) → GN → ReLU → AdaptAvgPool(1x1)
        FC(256→256) → ReLU → Dropout(0.1) → FC(256→embed_dim) → L2 norm

    GroupNorm instead of BatchNorm: identical behaviour in train and eval,
    no running statistics that can collapse word images to a single point.
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

# ─── NT-XENT LOSS (SimCLR) ───────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy Loss (SimCLR).
    Every sample in the batch serves as a negative for every other sample.
    Dense gradient signal — no margin to miscalibrate.

    Input: two batches of L2-normalised embeddings (emb_a, emb_b),
           each row is a positive pair.
    """
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a, emb_b):
        batch_size = emb_a.size(0)
        emb = torch.cat([emb_a, emb_b], dim=0)          # (2B, D)
        sim = torch.mm(emb, emb.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=emb.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Positive indices: i pairs with i+B and vice versa
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=emb.device),
            torch.arange(batch_size, device=emb.device),
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
    # NATURE
    ("night",    ["night","Nacht","nuit","noche","ночь","ليل","रात","夜","νύχτα","לילה","กลางคืน","gece","natt"]),
    ("water",    ["water","Wasser","eau","agua","вода","ماء","पानी","水","νερό","מים","น้ำ","su","vatten"]),
    ("fire",     ["fire","Feuer","feu","fuego","огонь","نار","आग","火","φωτιά","אש","ไฟ","ateş","eld"]),
    ("earth",    ["earth","Erde","terre","tierra","земля","أرض","पृथ्वी","地球","γη","אדמה","โลก","toprak","jord"]),
    ("air",      ["air","Luft","air","aire","воздух","هواء","हवा","空气","αέρας","אוויר","อากาศ","hava","luft"]),
    ("sun",      ["sun","Sonne","soleil","sol","солнце","شمس","सूरज","太阳","ήλιος","שמש","ดวงอาทิตย์","güneş","sol"]),
    ("moon",     ["moon","Mond","lune","luna","луна","قمر","चाँद","月","φεγγάρι","ירח","ดวงจันทร์","ay","måne"]),
    ("star",     ["star","Stern","étoile","estrella","звезда","نجمة","तारा","星","αστέρι","כוכב","ดาว","yıldız","stjärna"]),
    ("sky",      ["sky","Himmel","ciel","cielo","небо","سماء","आकाश","天空","ουρανός","שמיים","ท้องฟ้า","gökyüzü","himmel"]),
    ("sea",      ["sea","Meer","mer","mar","море","بحر","समुद्र","海","θάλασσα","ים","ทะเล","deniz","hav"]),
    ("river",    ["river","Fluss","rivière","río","река","نهر","नदी","河","ποτάμι","נהר","แม่น้ำ","nehir","flod"]),
    ("mountain", ["mountain","Berg","montagne","montaña","гора","جبل","पहाड़","山","βουνό","הר","ภูเขา","dağ","berg"]),
    ("forest",   ["forest","Wald","forêt","bosque","лес","غابة","जंगल","森林","δάσος","יער","ป่า","orman","skog"]),
    ("tree",     ["tree","Baum","arbre","árbol","дерево","شجرة","पेड़","树","δέντρο","עץ","ต้นไม้","ağaç","träd"]),
    ("flower",   ["flower","Blume","fleur","flor","цветок","زهرة","फूल","花","λουλούδι","פרח","ดอกไม้","çiçek","blomma"]),
    ("grass",    ["grass","Gras","herbe","hierba","трава","عشب","घास","草","χόρτο","עשב","หญ้า","çimen","gräs"]),
    ("stone",    ["stone","Stein","pierre","piedra","камень","حجر","पत्थर","石","πέτρα","אבן","หิน","taş","sten"]),
    ("sand",     ["sand","Sand","sable","arena","песок","رمل","रेत","沙","άμμος","חול","ทราย","kum","sand"]),
    ("snow",     ["snow","Schnee","neige","nieve","снег","ثلج","बर्फ","雪","χιόνι","שלג","หิมะ","kar","snö"]),
    ("rain",     ["rain","Regen","pluie","lluvia","дождь","مطر","बारिश","雨","βροχή","גשם","ฝน","yağmur","regn"]),
    ("wind",     ["wind","Wind","vent","viento","ветер","ريح","हवा","风","άνεμος","רוח","ลม","rüzgar","vind"]),
    ("cloud",    ["cloud","Wolke","nuage","nube","облако","سحابة","बादल","云","σύννεφο","ענן","เมฆ","bulut","moln"]),
    ("ice",      ["ice","Eis","glace","hielo","лёд","جليد","बर्फ","冰","πάγος","קרח","น้ำแข็ง","buz","is"]),
    ("lake",     ["lake","See","lac","lago","озеро","بحيرة","झील","湖","λίμνη","אגם","ทะเลสาบ","göl","sjö"]),
    # BODY
    ("hand",     ["hand","Hand","main","mano","рука","يد","हाथ","手","χέρι","יד","มือ","el","hand"]),
    ("eye",      ["eye","Auge","oeil","ojo","глаз","عين","आँख","眼","μάτι","עין","ตา","göz","öga"]),
    ("head",     ["head","Kopf","tête","cabeza","голова","رأس","सिर","头","κεφάλι","ראש","หัว","baş","huvud"]),
    ("heart",    ["heart","Herz","coeur","corazón","сердце","قلب","दिल","心","καρδιά","לב","หัวใจ","kalp","hjärta"]),
    ("mouth",    ["mouth","Mund","bouche","boca","рот","فم","मुँह","嘴","στόμα","פה","ปาก","ağız","mun"]),
    ("ear",      ["ear","Ohr","oreille","oreja","ухо","أذن","कान","耳","αυτί","אוזן","หู","kulak","öra"]),
    ("nose",     ["nose","Nase","nez","nariz","нос","أنف","नाक","鼻","μύτη","אף","จมูก","burun","näsa"]),
    ("foot",     ["foot","Fuß","pied","pie","нога","قدم","पाँव","脚","πόδι","רגל","เท้า","ayak","fot"]),
    ("blood",    ["blood","Blut","sang","sangre","кровь","دم","खून","血","αίμα","דם","เลือด","kan","blod"]),
    ("bone",     ["bone","Knochen","os","hueso","кость","عظم","हड्डी","骨","κόκκαλο","עצם","กระดูก","kemik","ben"]),
    ("skin",     ["skin","Haut","peau","piel","кожа","جلد","चमड़ी","皮肤","δέρμα","עור","ผิว","deri","hud"]),
    ("hair",     ["hair","Haar","cheveux","pelo","волосы","شعر","बाल","头发","μαλλιά","שיער","ผม","saç","hår"]),
    ("tooth",    ["tooth","Zahn","dent","diente","зуб","سن","दाँत","牙","δόντι","שן","ฟัน","diş","tand"]),
    ("finger",   ["finger","Finger","doigt","dedo","палец","إصبع","उँगली","手指","δάχτυλο","אצבע","นิ้ว","parmak","finger"]),
    ("leg",      ["leg","Bein","jambe","pierna","нога","ساق","टाँग","腿","πόδι","רגל","ขา","bacak","ben"]),
    ("arm",      ["arm","Arm","bras","brazo","рука","ذراع","बाँह","手臂","χέρι","זרוע","แขน","kol","arm"]),
    ("face",     ["face","Gesicht","visage","cara","лицо","وجه","चेहरा","脸","πρόσωπο","פנים","หน้า","yüz","ansikte"]),
    ("neck",     ["neck","Hals","cou","cuello","шея","رقبة","गर्दन","脖子","λαιμός","צוואר","คอ","boyun","hals"]),
    # FAMILY
    ("mother",   ["mother","Mutter","mère","madre","мать","أم","माँ","母亲","μητέρα","אמא","แม่","anne","mor"]),
    ("father",   ["father","Vater","père","padre","отец","أب","पिता","父亲","πατέρας","אבא","พ่อ","baba","far"]),
    ("child",    ["child","Kind","enfant","niño","ребёнок","طفل","बच्चा","孩子","παιδί","ילד","เด็ก","çocuk","barn"]),
    ("brother",  ["brother","Bruder","frère","hermano","брат","أخ","भाई","兄弟","αδελφός","אח","พี่ชาย","erkek kardeş","bror"]),
    ("sister",   ["sister","Schwester","soeur","hermana","сестра","أخت","बहन","姐妹","αδελφή","אחות","น้องสาว","kız kardeş","syster"]),
    ("son",      ["son","Sohn","fils","hijo","сын","ابن","बेटा","儿子","γιος","בן","ลูกชาย","oğul","son"]),
    ("daughter", ["daughter","Tochter","fille","hija","дочь","ابنة","बेटी","女儿","κόρη","בת","ลูกสาว","kız","dotter"]),
    ("family",   ["family","Familie","famille","familia","семья","عائلة","परिवार","家庭","οικογένεια","משפחה","ครอบครัว","aile","familj"]),
    ("friend",   ["friend","Freund","ami","amigo","друг","صديق","दोस्त","朋友","φίλος","חבר","เพื่อน","arkadaş","vän"]),
    ("man",      ["man","Mann","homme","hombre","мужчина","رجل","आदमी","男人","άντρας","גבר","ผู้ชาย","adam","man"]),
    ("woman",    ["woman","Frau","femme","mujer","женщина","امرأة","औरत","女人","γυναίκα","אישה","ผู้หญิง","kadın","kvinna"]),
    # TIME
    ("day",      ["day","Tag","jour","día","день","يوم","दिन","天","μέρα","יום","วัน","gün","dag"]),
    ("year",     ["year","Jahr","an","año","год","سنة","साल","年","χρόνος","שנה","ปี","yıl","år"]),
    ("time",     ["time","Zeit","temps","tiempo","время","وقت","समय","时间","χρόνος","זמן","เวลา","zaman","tid"]),
    ("hour",     ["hour","Stunde","heure","hora","час","ساعة","घंटा","小时","ώρα","שעה","ชั่วโมง","saat","timme"]),
    ("week",     ["week","Woche","semaine","semana","неделя","أسبوع","हफ्ता","星期","εβδομάδα","שבוע","สัปดาห์","hafta","vecka"]),
    ("morning",  ["morning","Morgen","matin","mañana","утро","صباح","सुबह","早上","πρωί","בוקר","เช้า","sabah","morgon"]),
    ("evening",  ["evening","Abend","soir","tarde","вечер","مساء","शाम","晚上","βράδυ","ערב","เย็น","akşam","kväll"]),
    ("today",    ["today","heute","aujourd'hui","hoy","сегодня","اليوم","आज","今天","σήμερα","היום","วันนี้","bugün","idag"]),
    ("old",      ["old","alt","vieux","viejo","старый","قديم","पुराना","旧","παλιός","ישן","เก่า","eski","gammal"]),
    ("new",      ["new","neu","nouveau","nuevo","новый","جديد","नया","新","νέος","חדש","ใหม่","yeni","ny"]),
    # ACTIONS
    ("eat",      ["eat","essen","manger","comer","есть","أكل","खाना","吃","τρώω","אכול","กิน","yemek","äta"]),
    ("drink",    ["drink","trinken","boire","beber","пить","شرب","पीना","喝","πίνω","שתות","ดื่ม","içmek","dricka"]),
    ("sleep",    ["sleep","schlafen","dormir","dormir","спать","نوم","सोना","睡觉","κοιμάμαι","לישון","นอน","uyumak","sova"]),
    ("walk",     ["walk","gehen","marcher","caminar","идти","مشي","चलना","走路","περπατώ","ללכת","เดิน","yürümek","gå"]),
    ("run",      ["run","rennen","courir","correr","бежать","جري","दौड़ना","跑","τρέχω","לרוץ","วิ่ง","koşmak","springa"]),
    ("see",      ["see","sehen","voir","ver","видеть","رأى","देखना","看","βλέπω","לראות","มองเห็น","görmek","se"]),
    ("hear",     ["hear","hören","entendre","oír","слышать","سمع","सुनना","听","ακούω","לשמוע","ได้ยิน","duymak","höra"]),
    ("speak",    ["speak","sprechen","parler","hablar","говорить","تكلم","बोलना","说话","μιλώ","לדבר","พูด","konuşmak","tala"]),
    ("know",     ["know","wissen","savoir","saber","знать","يعرف","जानना","知道","ξέρω","לדעת","รู้","bilmek","veta"]),
    ("give",     ["give","geben","donner","dar","давать","أعطى","देना","给","δίνω","לתת","ให้","vermek","ge"]),
    ("come",     ["come","kommen","venir","venir","приходить","جاء","आना","来","έρχομαι","לבוא","มา","gelmek","komma"]),
    ("go",       ["go","gehen","aller","ir","идти","ذهب","जाना","去","πηγαίνω","ללכת","ไป","gitmek","gå"]),
    ("make",     ["make","machen","faire","hacer","делать","صنع","बनाना","做","κάνω","לעשות","ทำ","yapmak","göra"]),
    ("want",     ["want","wollen","vouloir","querer","хотеть","أراد","चाहना","想要","θέλω","לרצות","ต้องการ","istemek","vilja"]),
    ("think",    ["think","denken","penser","pensar","думать","فكر","सोचना","想","σκέφτομαι","לחשוב","คิด","düşünmek","tänka"]),
    ("work",     ["work","arbeiten","travailler","trabajar","работать","عمل","काम","工作","εργάζομαι","לעבוד","ทำงาน","çalışmak","arbeta"]),
    ("live",     ["live","leben","vivre","vivir","жить","يعيش","जीना","活","ζω","לחיות","อยู่","yaşamak","leva"]),
    ("read",     ["read","lesen","lire","leer","читать","قرأ","पढ़ना","读","διαβάζω","לקרוא","อ่าน","okumak","läsa"]),
    ("write",    ["write","schreiben","écrire","escribir","писать","كتب","लिखना","写","γράφω","לכתוב","เขียน","yazmak","skriva"]),
    ("buy",      ["buy","kaufen","acheter","comprar","покупать","اشترى","खरीदना","买","αγοράζω","לקנות","ซื้อ","satın almak","köpa"]),
    # OBJECTS
    ("house",    ["house","Haus","maison","casa","дом","بيت","घर","家","σπίτι","בית","บ้าน","ev","hus"]),
    ("book",     ["book","Buch","livre","libro","книга","كتاب","किताب","书","βιβλίο","ספר","หนังสือ","kitap","bok"]),
    ("door",     ["door","Tür","porte","puerta","дверь","باب","दरवाजा","门","πόρτα","דלת","ประตู","kapı","dörr"]),
    ("window",   ["window","Fenster","fenêtre","ventana","окно","نافذة","खिड़की","窗户","παράθυρο","חלון","หน้าต่าง","pencere","fönster"]),
    ("table",    ["table","Tisch","table","mesa","стол","طاولة","मेज","桌子","τραπέζι","שולחן","โต๊ะ","masa","bord"]),
    ("bed",      ["bed","Bett","lit","cama","кровать","سرير","बिस्तर","床","κρεβάτι","מיטה","เตียง","yatak","säng"]),
    ("car",      ["car","Auto","voiture","coche","машина","سيارة","गाड़ी","汽车","αυτοκίνητο","מכונית","รถ","araba","bil"]),
    ("road",     ["road","Straße","route","camino","дорога","طريق","रास्ता","路","δρόμος","דרך","ถนน","yol","väg"]),
    ("money",    ["money","Geld","argent","dinero","деньги","مال","पैसा","钱","χρήματα","כסף","เงิน","para","pengar"]),
    ("food",     ["food","Essen","nourriture","comida","еда","طعام","खाना","食物","τροφή","אוכל","อาหาร","yemek","mat"]),
    ("bread",    ["bread","Brot","pain","pan","хлеб","خبز","रोटी","面包","ψωμί","לחם","ขนมปัง","ekmek","bröd"]),
    ("milk",     ["milk","Milch","lait","leche","молоко","حليب","दूध","牛奶","γάλα","חלב","นม","süt","mjölk"]),
    ("key",      ["key","Schlüssel","clé","llave","ключ","مفتاح","चाबी","钥匙","κλειδί","מפתח","กุญแจ","anahtar","nyckel"]),
    ("rope",     ["rope","Seil","corde","cuerda","верёвка","حبل","रस्सी","绳子","σχοινί","חבל","เชือก","ip","rep"]),
    ("clothes",  ["clothes","Kleidung","vêtements","ropa","одежда","ملابس","कपड़े","衣服","ρούχα","בגדים","เสื้อผ้า","kıyafet","kläder"]),
    ("ship",     ["ship","Schiff","navire","barco","корабль","سفينة","जहाज","船","πλοίο","אוניה","เรือ","gemi","skepp"]),
    ("bridge",   ["bridge","Brücke","pont","puente","мост","جسر","पुल","桥","γέφυρα","גשר","สะพาน","köprü","bro"]),
    # COLOURS
    ("red",      ["red","rot","rouge","rojo","красный","أحمر","लाल","红","κόκκινος","אדום","แดง","kırmızı","röd"]),
    ("blue",     ["blue","blau","bleu","azul","синий","أزرق","नीला","蓝","μπλε","כחול","น้ำเงิน","mavi","blå"]),
    ("green",    ["green","grün","vert","verde","зелёный","أخضر","हरा","绿","πράσινος","ירוק","เขียว","yeşil","grön"]),
    ("white",    ["white","weiß","blanc","blanco","белый","أبيض","सफेद","白","λευκός","לבן","ขาว","beyaz","vit"]),
    ("black",    ["black","schwarz","noir","negro","чёрный","أسود","काला","黑","μαύρος","שחור","ดำ","siyah","svart"]),
    ("yellow",   ["yellow","gelb","jaune","amarillo","жёлтый","أصفر","पीला","黄","κίτρινος","צהוב","เหลือง","sarı","gul"]),
    # NUMBERS
    ("one",      ["one","eins","un","uno","один","واحد","एक","一","ένα","אחד","หนึ่ง","bir","ett"]),
    ("two",      ["two","zwei","deux","dos","два","اثنان","दो","二","δύο","שתיים","สอง","iki","två"]),
    ("three",    ["three","drei","trois","tres","три","ثلاثة","तीन","三","τρία","שלוש","สาม","üç","tre"]),
    ("five",     ["five","fünf","cinq","cinco","пять","خمسة","पाँच","五","πέντε","חמש","ห้า","beş","fem"]),
    ("ten",      ["ten","zehn","dix","diez","десять","عشرة","दस","十","δέκα","עשר","สิบ","on","tio"]),
    # ABSTRACT
    ("love",     ["love","Liebe","amour","amor","любовь","حب","प्यार","爱","αγάπη","אהבה","ความรัก","aşk","kärlek"]),
    ("peace",    ["peace","Frieden","paix","paz","мир","سلام","शांति","和平","ειρήνη","שלום","สันติภาพ","barış","fred"]),
    ("war",      ["war","Krieg","guerre","guerra","война","حرب","युद्ध","战争","πόλεμος","מלחמה","สงคราม","savaş","krig"]),
    ("truth",    ["truth","Wahrheit","vérité","verdad","правда","حقيقة","सच","真相","αλήθεια","אמת","ความจริง","gerçek","sanning"]),
    ("life",     ["life","Leben","vie","vida","жизнь","حياة","जीवन","生命","ζωή","חיים","ชีวิต","hayat","liv"]),
    ("death",    ["death","Tod","mort","muerte","смерть","موت","मृत्यु","死亡","θάνατος","מוות","ความตาย","ölüm","död"]),
    ("freedom",  ["freedom","Freiheit","liberté","libertad","свобода","حرية","आजादी","自由","ελευθερία","חופש","อิสรภาพ","özgürlük","frihet"]),
    ("hope",     ["hope","Hoffnung","espoir","esperanza","надежда","أمل","उम्मीद","希望","ελπίδα","תקווה","ความหวัง","umut","hopp"]),
    ("happy",    ["happy","glücklich","heureux","feliz","счастливый","سعيد","खुश","快乐","ευτυχισμένος","שמח","มีความสุข","mutlu","lycklig"]),
    ("good",     ["good","gut","bon","bueno","хороший","جيد","अच्छा","好","καλός","טוב","ดี","iyi","bra"]),
    ("bad",      ["bad","schlecht","mauvais","malo","плохой","سيء","बुरا","坏","κακός","רע","แย่","kötü","dålig"]),
    ("big",      ["big","groß","grand","grande","большой","كبير","बड़ा","大","μεγάλος","גדול","ใหญ่","büyük","stor"]),
    ("small",    ["small","klein","petit","pequeño","маленький","صغير","छोटा","小","μικρός","קטן","เล็ก","küçük","liten"]),
    ("hot",      ["hot","heiß","chaud","caliente","горячий","حار","गर्म","热","ζεστός","חם","ร้อน","sıcak","het"]),
    ("cold",     ["cold","kalt","froid","frío","холодный","بارد","ठंडा","冷","κρύος","קר","เย็น","soğuk","kall"]),
    ("fast",     ["fast","schnell","rapide","rápido","быстрый","سريع","तेज","快","γρήγορος","מהיר","เร็ว","hızlı","snabb"]),
    ("long",     ["long","lang","long","largo","длинный","طويل","लंबा","长","μακρύς","ארוך","ยาว","uzun","lång"]),
    # GOVERNANCE
    ("king",     ["king","König","roi","rey","король","ملك","राजा","国王","βασιλιάς","מלך","กษัตริย์","kral","kung"]),
    ("law",      ["law","Gesetz","loi","ley","закон","قانون","कानून","法律","νόμος","חוק","กฎหมาย","kanun","lag"]),
    ("city",     ["city","Stadt","ville","ciudad","город","مدينة","शहर","城市","πόλη","עיר","เมือง","şehir","stad"]),
    ("country",  ["country","Land","pays","país","страна","بلد","देश","国家","χώρα","מדינה","ประเทศ","ülke","land"]),
    ("school",   ["school","Schule","école","escuela","школа","مدرسة","स्कूल","学校","σχολείο","בית ספר","โรงเรียน","okul","skola"]),
    ("doctor",   ["doctor","Arzt","médecin","médico","врач","طبيب","डॉक्टर","医生","γιατρός","רופא","หมอ","doktor","läkare"]),
    # GEOGRAPHY
    ("north",    ["north","Norden","nord","norte","север","شمال","उत्तर","北","βορράς","צפון","เหนือ","kuzey","norr"]),
    ("south",    ["south","Süden","sud","sur","юг","جنوب","दक्षिण","南","νότος","דרום","ใต้","güney","söder"]),
    ("east",     ["east","Osten","est","este","восток","شرق","पूर्व","东","ανατολή","מזרח","ตะวันออก","doğu","öster"]),
    ("west",     ["west","Westen","ouest","oeste","запад","غرب","पश्चिम","西","δύση","מערב","ตะวันตก","batı","väster"]),
    # ANIMALS
    ("dog",      ["dog","Hund","chien","perro","собака","كلب","कुत्ता","狗","σκύλος","כלב","สุนัข","köpek","hund"]),
    ("cat",      ["cat","Katze","chat","gato","кошка","قطة","बिल्ली","猫","γάτα","חתול","แมว","kedi","katt"]),
    ("bird",     ["bird","Vogel","oiseau","pájaro","птица","طائر","पक्षी","鸟","πουλί","ציפור","นก","kuş","fågel"]),
    ("fish",     ["fish","Fisch","poisson","pez","рыба","سمكة","मछली","鱼","ψάρι","דג","ปลา","balık","fisk"]),
    ("horse",    ["horse","Pferd","cheval","caballo","лошадь","حصان","घोड़ा","马","άλογο","סוס","ม้า","at","häst"]),
    ("lion",     ["lion","Löwe","lion","león","лев","أسد","शेर","狮子","λιοντάρι","אריה","สิงโต","aslan","lejon"]),
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
    Produces positive pairs for NT-Xent training.
    Two types:
    1. Same word, font size variation          → positive pair
    2. Same concept across languages           → positive pair

    No negative labels needed — NT-Xent treats all other pairs
    in the batch as negatives automatically.
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

        # Font size variation range scaled to IMG_H=96 (was 17-23 at 32px)
        font_sizes = [50, 54, 58, 62, 66, 70, 74]

        raw_pairs = []
        for _ in range(n_pairs // 5):
            word = random.choice(word_list)
            sa = random.choice(font_sizes)
            sb = random.choice(font_sizes)
            raw_pairs.append((word, sa, word, sb))
        for _ in range(n_pairs * 4 // 5):
            _, translations = random.choice(SEMANTIC_CONCEPTS)
            a = random.choice(translations)
            b = random.choice(translations)
            raw_pairs.append((a, FONT_SIZE, b, FONT_SIZE))
        random.shuffle(raw_pairs)

        print(f"  Pre-rendering {len(raw_pairs):,} pairs into RAM...")
        for wa, sa, wb, sb in tqdm(raw_pairs, desc="  rendering"):
            ta = word_to_tensor(wa, sa)
            tb = word_to_tensor(wb, sb)
            self.pairs.append((ta, tb))
        print(f"  Pre-rendering done.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

# ─── TRAINING ────────────────────────────────────────────────────────────────

def train(model, epochs=EPOCHS, n_pairs=5000, lr=LR):
    print(f"\n=== TRAINING on {DEVICE} ===")
    print(f"  Pairs: {n_pairs}  Batch: {BATCH_SIZE}  Epochs: {epochs}")

    dataset = WordPairDataset(n_pairs=n_pairs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=True)

    criterion = NTXentLoss(temperature=TEMPERATURE)
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
