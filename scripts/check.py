import os
from pathlib import Path
import yaml
from collections import Counter

DATA_YAML = "data.yaml"

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

print("DATA YAML:")
print(data)

names = data["names"]
print("\nCLASS NAMES:")
print(names)

dataset_path = Path(data.get("path", "."))
label_dirs = [
    dataset_path / "train" / "labels",
    dataset_path / "val" / "labels",
    dataset_path / "val" / "labels",
    dataset_path / "test" / "labels",
    dataset_path / "labels" / "train",
    dataset_path / "labels" / "val",
    dataset_path / "labels" / "test",
]

counter = Counter()

for label_dir in label_dirs:
    if not label_dir.exists():
        continue

    print(f"\nĐang kiểm tra: {label_dir}")

    for txt_file in label_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cls_id = int(float(line.split()[0]))
                    counter[cls_id] += 1

print("\nTHỐNG KÊ CLASS ID TRONG LABEL:")
for cls_id, count in sorted(counter.items()):
    name = names.get(cls_id, "UNKNOWN") if isinstance(names, dict) else names[cls_id]
    print(f"{cls_id} - {name}: {count}")