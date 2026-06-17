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
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>Wood Defect Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                margin: 0;
                padding: 0;
            }

            .container {
                width: 90%;
                max-width: 1100px;
                margin: 30px auto;
                background: white;
                padding: 25px;
                border-radius: 14px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.12);
            }

            h1 {
                text-align: center;
                color: #6b3f1d;
                margin-bottom: 10px;
            }

            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 25px;
            }

            .upload-box {
                border: 2px dashed #b88752;
                padding: 25px;
                text-align: center;
                border-radius: 12px;
                background: #fffaf4;
            }

            input[type="file"] {
                margin: 15px 0;
            }

            button {
                padding: 12px 24px;
                background: #8b5a2b;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
            }

            button:hover {
                background: #6f431f;
            }

            .result {
                margin-top: 30px;
                display: none;
            }

            .status {
                font-size: 22px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
                padding: 12px;
                border-radius: 8px;
            }

            .pass {
                background: #e7f8ec;
                color: #147a31;
            }

            .ng {
                background: #fdeaea;
                color: #b00020;
            }

            .images {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }

            .image-card {
                text-align: center;
            }

            .image-card img {
                max-width: 100%;
                border-radius: 10px;
                border: 1px solid #ddd;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 25px;
            }

            th, td {
                border: 1px solid #ddd;
                padding: 10px;
                text-align: center;
            }

            th {
                background: #8b5a2b;
                color: white;
            }

            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
                color: #8b5a2b;
                font-weight: bold;
            }

            @media (max-width: 768px) {
                .images {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Wood Surface Defect Detection</h1>
            <div class="subtitle">Nhận diện lỗi bề mặt gỗ bằng mô hình YOLOv8</div>

            <div class="upload-box">
                <h3>Chọn ảnh bề mặt gỗ cần kiểm tra</h3>
                <input type="file" id="fileInput" accept="image/*">
                <br>
                <button onclick="predict()">Nhận diện lỗi</button>
            </div>

            <div class="loading" id="loading">
                Đang xử lý ảnh, vui lòng chờ...
            </div>

            <div class="result" id="resultBox">
                <div id="qualityStatus" class="status"></div>

                <p><b>Tổng số lỗi phát hiện:</b> <span id="totalDefects"></span></p>

                <div class="images">
                    <div class="image-card">
                        <h3>Ảnh gốc</h3>
                        <img id="originalImage" src="">
                    </div>

                    <div class="image-card">
                        <h3>Ảnh sau khi nhận diện</h3>
                        <img id="resultImage" src="">
                    </div>
                </div>

                <table>
                    <thead>
                        <tr>
                            <th>STT</th>
                            <th>Loại lỗi</th>
                            <th>Độ tin cậy</th>
                            <th>x1</th>
                            <th>y1</th>
                            <th>x2</th>
                            <th>y2</th>
                        </tr>
                    </thead>
                    <tbody id="detectionTable"></tbody>
                </table>
            </div>
        </div>

        <script>
            async function predict() {
                const fileInput = document.getElementById("fileInput");
                const file = fileInput.files[0];

                if (!file) {
                    alert("Vui lòng chọn một ảnh trước.");
                    return;
                }

                const formData = new FormData();
                formData.append("file", file);

                document.getElementById("loading").style.display = "block";
                document.getElementById("resultBox").style.display = "none";

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        alert(error.detail || "Có lỗi xảy ra.");
                        return;
                    }

                    const data = await response.json();

                    document.getElementById("loading").style.display = "none";
                    document.getElementById("resultBox").style.display = "block";

                    document.getElementById("originalImage").src = data.original_image_url;
                    document.getElementById("resultImage").src = data.result_image_url;
                    document.getElementById("totalDefects").innerText = data.total_defects;

                    const statusDiv = document.getElementById("qualityStatus");
                    statusDiv.innerText = data.quality_status;

                    if (data.quality_status === "PASS") {
                        statusDiv.className = "status pass";
                    } else {
                        statusDiv.className = "status ng";
                    }

                    const table = document.getElementById("detectionTable");
                    table.innerHTML = "";

                    if (data.detections.length === 0) {
                        table.innerHTML = `
                            <tr>
                                <td colspan="7">Không phát hiện lỗi</td>
                            </tr>
                        `;
                    } else {
                        data.detections.forEach((det, index) => {
                            table.innerHTML += `
                                <tr>
                                    <td>${index + 1}</td>
                                    <td>${det.class_name}</td>
                                    <td>${det.confidence}</td>
                                    <td>${det.bbox.x1}</td>
                                    <td>${det.bbox.y1}</td>
                                    <td>${det.bbox.x2}</td>
                                    <td>${det.bbox.y2}</td>
                                </tr>
                            `;
                        });
                    }

                } catch (error) {
                    document.getElementById("loading").style.display = "none";
                    alert("Lỗi kết nối server.");
                    console.error(error);
                }
            }
        </script>
    </body>
    </html>
    """


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
async def predict(file: Annotated[UploadFile, File(...)]):
    if not check_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail="File không hợp lệ. Chỉ hỗ trợ jpg, jpeg, png, bmp, webp."
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
            conf=CONF_THRESHOLD,
            save=False,
            verbose=False
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi chạy model: {str(e)}"
        )

    detections = []

    for result in results:
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
        "detections": detections
    }