import os
import shutil
import random
from pathlib import Path

def split_dataset(train_dir, val_dir, val_ratio=0.2):
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    val_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        val_class_dir = val_dir / class_name
        val_class_dir.mkdir(parents=True, exist_ok=True)

        images = list(class_dir.glob("*.png"))
        random.shuffle(images)

        n_val = int(len(images) * val_ratio)
        val_samples = images[:n_val]

        for img_path in val_samples:
            shutil.move(str(img_path), str(val_class_dir / img_path.name))

        print(f"[✓] Класс {class_name}: {n_val} файлов перемещено в val/")

if __name__ == "__main__":
    split_dataset("spectrogram_dataset/train", "spectrogram_dataset/val", val_ratio=0.2)
