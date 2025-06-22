import os
from pathlib import Path

def check_yolo_dataset_structure(base_dir="EAGLE2"):
    splits = ["train", "valid", "test"]
    image_exts = {".jpg", ".jpeg", ".png"}
    label_ext = ".txt"
    errors = []

    print(f"\n🔍 Verifica del dataset YOLO in: {base_dir}\n{'-'*40}")

    for split in splits:
        img_dir = Path(base_dir) / split / "images"
        lbl_dir = Path(base_dir) / split / "labels"

        # Check if image and label directories exist
        if not img_dir.exists():
            errors.append(f"[{split}] ❌ Manca la cartella: {img_dir}")
            continue
        if not lbl_dir.exists():
            errors.append(f"[{split}] ❌ Manca la cartella: {lbl_dir}")
            continue

        # List image and label files
        img_files = list(img_dir.glob("*"))
        lbl_files = list(lbl_dir.glob("*.txt"))

        if not img_files:
            errors.append(f"[{split}] ⚠️ Nessuna immagine trovata in {img_dir}")
        if not lbl_files:
            errors.append(f"[{split}] ⚠️ Nessun file di label in {lbl_dir}")

        for lbl_file in lbl_files:
            with open(lbl_file, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        errors.append(f"[{split}] ❌ {lbl_file.name}, riga {i+1}: formato errato")
                        continue
                    try:
                        cls, x, y, w, h = map(float, parts)
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            errors.append(f"[{split}] ⚠️ {lbl_file.name}, riga {i+1}: valori fuori range [0, 1]")
                    except ValueError:
                        errors.append(f"[{split}] ❌ {lbl_file.name}, riga {i+1}: non numerico")

    # Report finale
    if errors:
        print("❌ Problemi trovati:\n" + "\n".join(errors))
    else:
        print("✅ Tutto ok! Il dataset è pronto per l'uso con Ultralytics YOLO.")

# Esegui lo script
check_yolo_dataset_structure("EAGLE2")
