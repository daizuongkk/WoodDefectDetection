import cv2
import numpy as np
import os
from pathlib import Path

# Cấu hình đường dẫn
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

# Tên các loại khiếm phẩm
CLASS_NAMES = {
    0: "scratch",
    1: "crack",
    2: "bubble",
    3: "hole"
}

# Màu sắc cho mỗi lớp (BGR format)
CLASS_COLORS = {
    0: (0, 255, 255),      # Vàng (scratch)
    1: (0, 0, 255),        # Đỏ (crack)
    2: (255, 0, 0),        # Xanh (bubble)
    3: (255, 255, 0)       # Xanh nhạt (hole)
}

def read_yolo_labels(label_file):
    """
    Đọc nhãn YOLO từ file .txt
    Format: class_id center_x center_y width height (tất cả là giá trị normalized)
    """
    boxes = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    boxes.append({
                        'class_id': class_id,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height
                    })
    return boxes

def convert_yolo_to_bbox(box, img_width, img_height):
    """
    Chuyển đổi tọa độ YOLO (normalized) sang tọa độ pixel
    """
    center_x = box['center_x'] * img_width
    center_y = box['center_y'] * img_height
    width = box['width'] * img_width
    height = box['height'] * img_height
    
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)
    
    return x1, y1, x2, y2

def draw_boxes_on_image(image, boxes):
    """
    Vẽ bounding boxes và labels lên ảnh
    """
    img_height, img_width = image.shape[:2]
    
    for box in boxes:
        class_id = box['class_id']
        x1, y1, x2, y2 = convert_yolo_to_bbox(box, img_width, img_height)
        
        # Đảm bảo tọa độ nằm trong phạm vi ảnh
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        # Vẽ hộp
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ nhãn
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        label_text = class_name
        
        # Lấy kích thước text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
        
        # Vẽ background cho text
        cv2.rectangle(image, (x1, y1 - text_size[1] - 5), 
                     (x1 + text_size[0] + 5, y1), color, -1)
        
        # Vẽ text
        cv2.putText(image, label_text, (x1 + 2, y1 - 3), 
                   font, font_scale, (255, 255, 255), thickness)
    
    return image

def resize_image(image, target_height=400):
    """
    Thay đổi kích thước ảnh theo chiều cao mục tiêu (giữ tỉ lệ)
    """
    height, width = image.shape[:2]
    ratio = target_height / height
    new_width = int(width * ratio)
    return cv2.resize(image, (new_width, target_height))

def display_sample_images():
    """
    Hiển thị 4 ảnh mẫu theo hàng ngang với bounding boxes
    """
    # Lấy danh sách các file ảnh
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(image_files) < 4:
        print(f"Chỉ tìm thấy {len(image_files)} ảnh, cần ít nhất 4 ảnh")
        return
    
    # Chọn 4 ảnh đầu tiên
    selected_images = image_files[:4]
    
    processed_images = []
    
    for img_file in selected_images:
        img_path = os.path.join(IMAGES_DIR, img_file)
        label_path = os.path.join(LABELS_DIR, img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        
        # Đọc ảnh
        image = cv2.imread(img_path)
        if image is None:
            print(f"Không thể đọc ảnh: {img_path}")
            continue
        
        # Đọc nhãn
        boxes = read_yolo_labels(label_path)
        
        # Vẽ boxes lên ảnh
        image_with_boxes = draw_boxes_on_image(image.copy(), boxes)
        
        # Thay đổi kích thước
        resized_image = resize_image(image_with_boxes, target_height=400)
        processed_images.append(resized_image)
    
    if len(processed_images) == 0:
        print("Không thể xử lý bất kỳ ảnh nào")
        return
    
    # Nếu chỉ có ít hơn 4 ảnh, thêm ảnh trống
    while len(processed_images) < 4:
        processed_images.append(np.ones((400, 400, 3), dtype=np.uint8) * 200)
    
    # Ghép 4 ảnh theo hàng ngang
    # Đảm bảo tất cả ảnh có cùng chiều cao
    target_height = 400
    resized_images = []
    for img in processed_images[:4]:
        h = img.shape[0]
        if h != target_height:
            ratio = target_height / h
            w = int(img.shape[1] * ratio)
            img = cv2.resize(img, (w, target_height))
        resized_images.append(img)
    
    # Ghép ảnh theo chiều ngang
    combined_image = cv2.hconcat(resized_images)
    
    # Lưu ảnh
    output_path = "sample_images_combined.jpg"
    cv2.imwrite(output_path, combined_image)
    print(f"Ảnh đã lưu tại: {output_path}")
    
    # Hiển thị ảnh
    cv2.imshow("4 Sample Images with Bounding Boxes", combined_image)
    print("Nhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_sample_images()
