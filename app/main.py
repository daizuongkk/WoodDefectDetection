import os
import uuid
import shutil
from typing import Annotated
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

MODEL_PATH = "runs/detect/wood_defect_yolov8n_grouped-2/weights/best.pt"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Wood Surface Defect Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(MODEL_PATH)


def classify_quality(detections):
    if len(detections) == 0:
        return "PASS"

    serious_classes = ["crack", "hole"]

    for detection in detections:
        if detection["class_name"] in serious_classes and detection["confidence"] >= 0.5:
            return "NG - Serious Defect"

    return "NG - Minor Defect"


@app.get("/")
def health_check():
    return {
        "message": "Wood Surface Defect Detection API is running"
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]):
    ext = file.filename.split(".")[-1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    image_path = os.path.join(UPLOAD_DIR, filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25
    )

    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })

    quality_status = classify_quality(detections)

    return {
        "filename": filename,
        "total_defects": len(detections),
        "quality_status": quality_status,
        "detections": detections
    }