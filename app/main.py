import os
import uuid
import shutil
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO


# =========================
# CẤU HÌNH
# =========================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "../runs/detect/wood_defect_yolov8s-3/weights/best.pt"

UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
RESULT_DIR = STATIC_DIR / "results"

UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

CONF_THRESHOLD = 0.25
IMG_SIZE = 640


# =========================
# KHỞI TẠO APP
# =========================

app = FastAPI(title="Wood Surface Defect Detection Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =========================
# LOAD MODEL
# =========================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))


# =========================
# HÀM XỬ LÝ
# =========================

def classify_quality(detections):
    """
    Phân loại chất lượng sản phẩm:
    - Không có lỗi: PASS
    - Có crack hoặc hole confidence >= 0.5: NG - Serious Defect
    - Còn lại: NG - Minor Defect
    """

    if len(detections) == 0:
        return "PASS"

    serious_classes = {
        "crack",
        "hole",
        "vết nứt",
        "vet nut",
        "lỗ",
        "lo",
        "dent"
    }

    for detection in detections:
        class_name = detection["class_name"].lower().strip()
        confidence = detection["confidence"]

        if class_name in serious_classes and confidence >= 0.5:
            return "NG - Serious Defect"

    return "NG - Minor Defect"


def check_file_extension(filename: str):
    if "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS


# =========================
# GIAO DIỆN WEB
# =========================

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


# =========================
# API KIỂM TRA SERVER
# =========================

@app.get("/api/health")
def health_check():
    return {
        "message": "Wood Surface Defect Detection API is running",
        "model_path": str(MODEL_PATH),
        "classes": model.names
    }


# =========================
# API DỰ ĐOÁN
# =========================

@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File(...)],
    conf: float = 0.25,
    iou: float = 0.45
):
    if not check_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail="File không hợp lệ. Chỉ hỗ trợ ảnh dạng: " + ", ".join(ALLOWED_EXTENSIONS)
        )

    ext = file.filename.rsplit(".", 1)[-1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    image_path = UPLOAD_DIR / filename

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=conf,
            iou=iou,
            save=False,
            verbose=False
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi chạy model: {str(e)}"
        )

    detections = []
    speed_metrics = {
        "preprocess": 0.0,
        "inference": 0.0,
        "postprocess": 0.0
    }

    if len(results) > 0:
        result = results[0]
        speed_metrics = {
            "preprocess": round(result.speed.get("preprocess", 0.0), 2),
            "inference": round(result.speed.get("inference", 0.0), 2),
            "postprocess": round(result.speed.get("postprocess", 0.0), 2)
        }
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = result.names.get(cls_id, str(cls_id))

            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })

    quality_status = classify_quality(detections)

    # Lưu ảnh đã vẽ bounding box
    result_filename = f"{Path(filename).stem}_result.jpg"
    result_image_path = RESULT_DIR / result_filename

    if len(results) > 0:
        annotated_image = results[0].plot()
        cv2.imwrite(str(result_image_path), annotated_image)
    else:
        original_image = cv2.imread(str(image_path))
        cv2.imwrite(str(result_image_path), original_image)

    return {
        "filename": filename,
        "total_defects": len(detections),
        "quality_status": quality_status,
        "original_image_url": f"/uploads/{filename}",
        "result_image_url": f"/static/results/{result_filename}",
        "detections": detections,
        "speed": speed_metrics
    }