from ultralytics import YOLO

model = YOLO("yolov8s.pt")
if __name__ == '__main__':

	model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    workers=4,
    patience=30,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    cos_lr=True,
    hsv_h=0.005,
    hsv_s=0.2,
    hsv_v=0.15,
    degrees=2,
    translate=0.03,
    scale=0.2,
    fliplr=0.3,
    mosaic=0.2,
    mixup=0.0,
    name="wood_defect_yolov8s"
)