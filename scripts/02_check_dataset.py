import os
from collections import Counter

DATASET_DIR = "dataset"
NUM_CLASSES = 4


def check_split(split):
    image_dir = os.path.join(DATASET_DIR, "images", split)
    label_dir = os.path.join(DATASET_DIR, "labels", split)

    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)

    print(f"\n===== {split.upper()} =====")
    print("Images:", len(images))
    print("Labels:", len(labels))

    class_counter = Counter()
    invalid_files = []

    for label_file in labels:
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, label_file)

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            parts = line.strip().split()

            if len(parts) == 0:
                continue

            if len(parts) != 5:
                invalid_files.append((label_file, line_idx, "Invalid format"))
                continue

            try:
                cls_id = int(parts[0])
                bbox = list(map(float, parts[1:]))
            except ValueError:
                invalid_files.append((label_file, line_idx, "Cannot parse value"))
                continue

            if cls_id < 0 or cls_id >= NUM_CLASSES:
                invalid_files.append((label_file, line_idx, f"Invalid class {cls_id}"))

            for value in bbox:
                if value < 0 or value > 1:
                    invalid_files.append((label_file, line_idx, f"Invalid bbox {bbox}"))

            class_counter[cls_id] += 1

    print("Class distribution:", dict(class_counter))

    if invalid_files:
        print("Invalid labels found:")
        for item in invalid_files[:20]:
            print(item)
    else:
        print("All labels are valid!")


def main():
    for split in ["train", "val", "test"]:
        check_split(split)


if __name__ == "__main__":
    main()