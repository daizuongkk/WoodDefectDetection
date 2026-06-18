# 🪵 Wood Surface Defect Detection (Phát Hiện Lỗi Bề Mặt Gỗ)

Dự án sử dụng mô hình học sâu **YOLOv8** kết hợp với framework **FastAPI** để phát hiện và phân loại các khuyết tật trên bề mặt gỗ (lỗ, bong bóng, vết nứt, vết xước) trong thời gian thực thông qua giao diện Web thân thiện.

---

## 🚀 Các Tính Năng Chính
* **Phát hiện đa lớp khuyết tật**: Hỗ trợ 4 nhóm lỗi phổ biến: `hole` (lỗ), `bubble` (bong bóng), `crack` (vết nứt), và `scratch` (vết xước).
* **Giao diện Web trực quan**: Cho phép tải ảnh lên, tùy chỉnh ngưỡng tin cậy (Confidence) và ngưỡng trùng lắp (IoU), hiển thị kết quả khoanh vùng lỗi tức thời.
* **Đánh giá chất lượng tự động**: Phân loại chất lượng gỗ thành `PASS` (Đạt chất lượng), `NG - Minor Defect` (Lỗi nhẹ), hoặc `NG - Serious Defect` (Lỗi nghiêm trọng).
* **Hiệu suất thời gian thực**: Thống kê thời gian tiền xử lý, suy luận (inference), và hậu xử lý cho mỗi bức ảnh.
* **Tích hợp sẵn Docker**: Dễ dàng đóng gói và triển khai trên mọi môi trường.

---

## 🛠️ Yêu Cầu Hệ Thống
* **Hệ điều hành**: Windows / Linux / macOS
* **Python**: Phiên bản `3.8` đến `3.11` (Khuyên dùng `3.10` hoặc `3.11`)
* **GPU**: Hỗ trợ CUDA (không bắt buộc, nhưng khuyến khích để huấn luyện và suy luận nhanh hơn)

---

## 📦 Hướng Dẫn Cài Đặt

### 1. Chuẩn bị môi trường
Mở terminal tại thư mục dự án và khởi tạo môi trường ảo Python:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows)
.\venv\Scripts\activate

# Kích hoạt môi trường ảo (Linux/macOS)
source venv/bin/activate
```

### 2. Cài đặt các thư viện phụ thuộc
Cập nhật `pip` và cài đặt các thư viện được liệt kê trong [requirements.txt](file:///d:/code/Project/WoodDefectDetection/requirements.txt):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Hướng Dẫn Chạy Web Application

Ứng dụng web được xây dựng bằng **FastAPI** (Backend) kết hợp **HTML/CSS/JS** (Frontend).

### Cách 1: Chạy trực tiếp trên máy local
Chạy lệnh uvicorn từ thư mục gốc của dự án:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

* **Truy cập Giao diện Web**: Mở trình duyệt và truy cập địa chỉ `http://127.0.0.1:8000`
* **Tài liệu API (Swagger UI)**: Truy cập `http://127.0.0.1:8000/docs` để test thử API
* **Kiểm tra trạng thái server**: Truy cập `http://127.0.0.1:8000/api/health`

### Cách 2: Chạy thông qua Docker
Nếu máy tính của bạn đã cài đặt Docker, bạn có thể build và chạy ứng dụng mà không cần cài đặt Python thủ công:

```bash
# 1. Build Docker Image
docker build -t wood-defect-app .

# 2. Chạy Docker Container (Chạy ở cổng 80 hoặc đổi sang cổng mong muốn)
docker run -d -p 8000:80 --name wood-defect-container wood-defect-app
```
Sau đó, truy cập giao diện tại `http://127.0.0.1:8000`.

---

## 🏋️ Huấn Luyện và Chạy Thử Nghiệm Mô Hình Offline

Dự án cung cấp sẵn một số script để huấn luyện mô hình YOLOv8 trên tập dữ liệu của bạn:

### 1. Cấu trúc dữ liệu huấn luyện
Đảm bảo cấu trúc thư mục dữ liệu tuân theo cấu hình trong file [data.yaml](file:///d:/code/Project/WoodDefectDetection/data.yaml):
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 2. Huấn luyện mô hình YOLOv8
Bạn có thể lựa chọn chạy huấn luyện phiên bản YOLOv8 Nano hoặc Small:

```bash
# Huấn luyện với YOLOv8 Nano (nhanh hơn, nhẹ hơn)
python train_yolov8n.py

# Huấn luyện với YOLOv8 Small (chính xác hơn, nặng hơn)
python train_yolov8s.py
```
*Kết quả huấn luyện (trọng số mô hình `best.pt`, đồ thị đánh giá) sẽ được lưu tại thư mục `runs/detect/`.*

### 3. Suy luận (Prediction) thử nghiệm offline
Để kiểm tra dự đoán nhanh trên một tệp ảnh bằng CLI, bạn hãy chạy file [predict.py](file:///d:/code/Project/WoodDefectDetection/predict.py):

```bash
python predict.py
```
*Ảnh kết quả kèm bounding box khoanh vùng lỗi sẽ được lưu vào thư mục `runs/detect/wood_defect_predictions/`.*

### 4. Đánh giá mô hình
Đánh giá chất lượng mô hình sau huấn luyện trên tập validation hoặc test:

```bash
python evaluate.py
```

---

## 📂 Sơ Đồ Cấu Trúc Dự Án
```
WoodDefectDetection/
├── app/
│   ├── main.py          # Backend API FastAPI
│   ├── index.html       # Frontend UI của ứng dụng
│   ├── static/          # Chứa các file tĩnh và thư mục chứa ảnh kết quả
│   └── uploads/         # Chứa các file ảnh được người dùng upload lên
├── dataset/             # Thư mục chứa dữ liệu hình ảnh & nhãn (YOLO format)
├── runs/                # Chứa kết quả huấn luyện mô hình và dự đoán
├── data.yaml            # Cấu hình đường dẫn dữ liệu và phân lớp
├── requirements.txt     # Danh sách các thư viện cần cài đặt
├── Dockerfile           # Docker configuration để deploy ứng dụng nhanh
├── train_yolov8n.py     # Script huấn luyện YOLOv8 Nano
├── train_yolov8s.py     # Script huấn luyện YOLOv8 Small
├── predict.py           # Script chạy suy đoán offline trên ảnh mẫu
└── README.md            # Tài liệu hướng dẫn sử dụng (File này)
```

---

## 📝 Các Lớp Nhãn Khuyết Tật (Dataset Classes)
Mô hình hỗ trợ phát hiện các lớp nhãn sau:
1. `0: hole` - Lỗ sâu, lỗ đục thủng trên bề mặt gỗ.
2. `1: bubble` - Bong bóng khí hoặc vết rộp trên bề mặt sơn/phủ.
3. `2: crack` - Vết nứt dọc hoặc ngang theo vân gỗ.
4. `3: scratch` - Vết xước do cọ xát cơ học.
