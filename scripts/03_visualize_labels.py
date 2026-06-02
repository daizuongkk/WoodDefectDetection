import os
import random
import cv2

DATASET_DIR = "dataset"
SPLIT = "train"

CLASS_NAMES = ["scratch", "crack", "bubble", "hole"]

IMAGE_DIR = os.path.join(DATASET_DIR, "images", SPLIT)
LABEL_DIR = os.path.join(DATASET_DIR, "labels", SPLIT)
OUTPUT_DIR = "visualized"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_yolo_boxes(image_path, label_path, output_path):
    img = cv2.imread(image_path)

    if img is None:
        return

    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            label = CLASS_NAMES[cls_id]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imwrite(output_path, img)


def main():
    images = os.listdir(IMAGE_DIR)
    sample_images = random.sample(images, min(50, len(images)))

    for img_file in sample_images:
        name = os.path.splitext(img_file)[0]

        image_path = os.path.join(IMAGE_DIR, img_file)
        label_path = os.path.join(LABEL_DIR, name + ".txt")
        output_path = os.path.join(OUTPUT_DIR, img_file)

        draw_yolo_boxes(image_path, label_path, output_path)

    print(f"Saved visualized images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()