from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO("yolov8n.pt")

    model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    workers=4,
    patience=20,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    cos_lr=True,

    # Dataset đã có augmentation sẵn nên để nhẹ
    hsv_h=0.005,
    hsv_s=0.2,
    hsv_v=0.15,
    degrees=2,
    translate=0.03,
    scale=0.2,
    fliplr=0.3,
    mosaic=0.2,
    mixup=0.0,

    name="wood_defect_yolov8n_grouped"
)