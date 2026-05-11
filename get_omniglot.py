"""
Download Omniglot and prepare for vwe.py add.

Downloads images_background.zip from GitHub and unpacks one or more
alphabets into data/<name>/images/ ready to use.

Usage:
    python get_omniglot.py                      # list available alphabets
    python get_omniglot.py --alphabet Tifinagh  # prepare one alphabet
    python get_omniglot.py --all                # prepare all 30 alphabets
"""

import argparse
import os
import zipfile
import urllib.request
import shutil

ZIP_URL  = "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip"
ZIP_PATH = "omniglot_background.zip"
UNPACK   = "omniglot_raw"
OUT_DIR  = "data"


def download_zip():
    if os.path.exists(ZIP_PATH):
        print(f"Already downloaded: {ZIP_PATH}")
        return
    print(f"Downloading {ZIP_URL} ...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print(f"Saved: {ZIP_PATH}")


def unpack_zip():
    if os.path.exists(UNPACK):
        print(f"Already unpacked: {UNPACK}/")
        return
    print(f"Unpacking {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(UNPACK)
    print(f"Unpacked to: {UNPACK}/")


def list_alphabets():
    root = os.path.join(UNPACK, "images_background")
    if not os.path.exists(root):
        unpack_zip()
    alphabets = sorted(os.listdir(root))
    print(f"\n{len(alphabets)} alphabets available:\n")
    for a in alphabets:
        chars = os.listdir(os.path.join(root, a))
        images = sum(
            len(os.listdir(os.path.join(root, a, c)))
            for c in chars
            if os.path.isdir(os.path.join(root, a, c))
        )
        print(f"  {a:<35} {len(chars):>3} chars  {images:>5} images")


def prepare_alphabet(alphabet_name):
    root = os.path.join(UNPACK, "images_background")
    src  = os.path.join(root, alphabet_name)

    if not os.path.exists(src):
        candidates = [a for a in os.listdir(root)
                      if alphabet_name.lower() in a.lower()]
        if not candidates:
            print(f"ERROR: '{alphabet_name}' not found.")
            print("Run without arguments to list available alphabets.")
            return
        alphabet_name = candidates[0]
        src = os.path.join(root, alphabet_name)
        print(f"Matched: {alphabet_name}")

    # Flatten all character images into data/<name>/images/
    name     = alphabet_name.replace(" ", "_").lower()
    out_imgs = os.path.join(OUT_DIR, name, "images")
    os.makedirs(out_imgs, exist_ok=True)

    count = 0
    for char_dir in sorted(os.listdir(src)):
        char_path = os.path.join(src, char_dir)
        if not os.path.isdir(char_path):
            continue
        for fname in sorted(os.listdir(char_path)):
            if not fname.lower().endswith(".png"):
                continue
            dst = os.path.join(out_imgs, f"{char_dir}_{fname}")
            shutil.copy2(os.path.join(char_path, fname), dst)
            count += 1

    print(f"Prepared: {out_imgs}/  ({count} images)")
    print()
    print(f"Run fine-tuning with:")
    print(f"  python vwe.py add --name {name} --script {name} --images {out_imgs}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Omniglot for vwe.py")
    parser.add_argument("--alphabet", default=None, help="Alphabet name to prepare")
    parser.add_argument("--all",      action="store_true", help="Prepare all alphabets")
    args = parser.parse_args()

    download_zip()
    unpack_zip()

    if args.all:
        root = os.path.join(UNPACK, "images_background")
        for a in sorted(os.listdir(root)):
            prepare_alphabet(a)
    elif args.alphabet:
        prepare_alphabet(args.alphabet)
    else:
        list_alphabets()


if __name__ == "__main__":
    main()
