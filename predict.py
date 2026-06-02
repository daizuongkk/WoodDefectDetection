from ultralytics import YOLO

MODEL_PATH = "runs/detect/wood_defect_yolov8s/weights/best.pt"
SOURCE = "000020.jpg"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=SOURCE,
    imgsz=640,
    conf=0.25,
    save=True,
    name="wood_defect_predictions"
)

for result in results:
    print("\nImage:", result.path)

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        print({
            "class": model.names[cls_id],
            "confidence": round(conf, 4),
            "bbox": [round(x, 2) for x in xyxy]
        })