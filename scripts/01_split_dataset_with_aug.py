import os
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict

ROOT_DIR = Path(".")
OUTPUT_DIR = Path("dataset")
ORIGINAL_DIR = "OriginalDataset"

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


def normalize_id(filename: str) -> str:
    stem = Path(filename).stem
    digits = "".join(ch for ch in stem if ch.isdigit())

    if digits:
        return str(int(digits))

    return stem.lower()


def reset_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def is_image(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in IMG_EXTS


def find_dataset_dirs():
    dataset_dirs = []

    for item in ROOT_DIR.iterdir():
        if not item.is_dir():
            continue

        if item.name in [
            "dataset",
            "scripts",
            "app",
            "runs",
            "uploads",
            "visualized",
            "__pycache__",
        ]:
            continue

        images_dir = item / "Images"
        labels_dir = item / "Labels"

        if images_dir.exists() and labels_dir.exists():
            dataset_dirs.append(item.name)

    if ORIGINAL_DIR not in dataset_dirs:
        raise FileNotFoundError(f"Không tìm thấy {ORIGINAL_DIR}/Images và {ORIGINAL_DIR}/Labels")

    return dataset_dirs


def get_original_images():
    image_dir = ROOT_DIR / ORIGINAL_DIR / "Images"

    images = [
        f.name for f in image_dir.iterdir()
        if f.is_file() and is_image(f.name)
    ]

    return sorted(images)


def read_classes_from_label(label_path: Path):
    classes = set()

    if not label_path.exists():
        return classes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) == 5:
                try:
                    classes.add(int(parts[0]))
                except ValueError:
                    pass

    return classes


def get_group_classes(original_img_file: str):
    name = Path(original_img_file).stem
    label_path = ROOT_DIR / ORIGINAL_DIR / "Labels" / f"{name}.txt"
    return read_classes_from_label(label_path)


def make_multilabel_stratified_split(original_images):
    """
    Split đơn giản nhưng có cân bằng class tương đối:
    - Ưu tiên ảnh có nhiều class trước
    - Đưa vào split đang thiếu class đó nhất
    """
    random.seed(SEED)

    items = []

    for img in original_images:
        group_id = normalize_id(img)
        classes = get_group_classes(img)

        items.append({
            "img": img,
            "id": group_id,
            "classes": classes
        })

    random.shuffle(items)

    total = len(items)
    target_counts = {
        "train": round(total * TRAIN_RATIO),
        "val": round(total * VAL_RATIO),
        "test": total - round(total * TRAIN_RATIO) - round(total * VAL_RATIO)
    }

    split_items = {
        "train": [],
        "val": [],
        "test": []
    }

    split_class_counter = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter()
    }

    # Ảnh có nhiều class được chia trước
    items.sort(key=lambda x: len(x["classes"]), reverse=True)

    for item in items:
        candidate_splits = [
            s for s in ["train", "val", "test"]
            if len(split_items[s]) < target_counts[s]
        ]

        if not candidate_splits:
            candidate_splits = ["train", "val", "test"]

        best_split = None
        best_score = None

        for split in candidate_splits:
            size_ratio = len(split_items[split]) / max(target_counts[split], 1)

            class_score = 0
            for cls in item["classes"]:
                class_score += split_class_counter[split][cls]

            score = size_ratio * 10 + class_score

            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        split_items[best_split].append(item)

        for cls in item["classes"]:
            split_class_counter[best_split][cls] += 1

    return split_items


def index_augmented_images(dataset_dirs):
    """
    Tạo index:
    {
      "BrightnessVariation": {
         "1": ["00001.jpg", "00001_xxx.jpg"]
      }
    }
    """
    index = {}

    for dataset_dir in dataset_dirs:
        image_dir = ROOT_DIR / dataset_dir / "Images"

        id_map = defaultdict(list)

        for file in image_dir.iterdir():
            if file.is_file() and is_image(file.name):
                img_id = normalize_id(file.name)
                id_map[img_id].append(file.name)

        index[dataset_dir] = id_map

    return index


def find_label_path(dataset_dir: str, img_file: str, original_img_file: str):
    aug_name = Path(img_file).stem
    original_name = Path(original_img_file).stem

    candidates = [
        ROOT_DIR / dataset_dir / "Labels" / f"{aug_name}.txt",
        ROOT_DIR / dataset_dir / "Labels" / f"{normalize_id(img_file).zfill(5)}.txt",
        ROOT_DIR / dataset_dir / "Labels" / f"{normalize_id(img_file).zfill(6)}.txt",
        ROOT_DIR / ORIGINAL_DIR / "Labels" / f"{original_name}.txt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def copy_pair(dataset_dir: str, img_file: str, original_img_file: str, split: str):
    img_src = ROOT_DIR / dataset_dir / "Images" / img_file

    if not img_src.exists():
        return False

    label_src = find_label_path(dataset_dir, img_file, original_img_file)

    img_stem = Path(img_file).stem
    img_ext = Path(img_file).suffix.lower()

    prefix = dataset_dir.lower() + "_"

    img_dst_name = f"{prefix}{img_stem}{img_ext}"
    label_dst_name = f"{prefix}{img_stem}.txt"

    img_dst = OUTPUT_DIR / "images" / split / img_dst_name
    label_dst = OUTPUT_DIR / "labels" / split / label_dst_name

    shutil.copy(img_src, img_dst)

    if label_src:
        shutil.copy(label_src, label_dst)
    else:
        label_dst.write_text("", encoding="utf-8")

    return True


def check_output():
    print("\n===== FINAL DATASET CHECK =====")

    for split in ["train", "val", "test"]:
        image_dir = OUTPUT_DIR / "images" / split
        label_dir = OUTPUT_DIR / "labels" / split

        images = list(image_dir.glob("*"))
        labels = list(label_dir.glob("*.txt"))

        counter = Counter()

        for label in labels:
            with open(label, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) == 5:
                        try:
                            counter[int(parts[0])] += 1
                        except ValueError:
                            pass

        print(f"\n{split.upper()}")
        print("Images:", len(images))
        print("Labels:", len(labels))
        print("Class distribution:", dict(counter))


def main():
    reset_output()

    dataset_dirs = find_dataset_dirs()
    original_images = get_original_images()

    print("Detected dataset folders:")
    for d in dataset_dirs:
        print("-", d)

    print("\nOriginal images:", len(original_images))

    split_items = make_multilabel_stratified_split(original_images)
    aug_index = index_augmented_images(dataset_dirs)

    copied_counter = Counter()

    for split, items in split_items.items():
        for item in items:
            original_img_file = item["img"]
            group_id = item["id"]

            for dataset_dir in dataset_dirs:
                matched_images = aug_index[dataset_dir].get(group_id, [])

                for img_file in matched_images:
                    copied = copy_pair(
                        dataset_dir=dataset_dir,
                        img_file=img_file,
                        original_img_file=original_img_file,
                        split=split
                    )

                    if copied:
                        copied_counter[(split, dataset_dir)] += 1

    print("\n===== COPY SUMMARY =====")
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}")
        total = 0

        for dataset_dir in dataset_dirs:
            count = copied_counter[(split, dataset_dir)]
            total += count
            print(f"{dataset_dir}: {count}")

        print("Total:", total)

    check_output()


if __name__ == "__main__":
    main()