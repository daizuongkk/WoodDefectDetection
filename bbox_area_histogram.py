import os
import numpy as np
import matplotlib.pyplot as plt

# ===================== CẤU HÌNH ĐƯỜNG DẪN =====================
DATASET_DIR = "dataset"
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

# ===================== TÊN CÁC LOẠI KHIẾM PHẨM =====================
CLASS_NAMES = {
    0: "Scratch",
    1: "Crack",
    2: "Bubble",
    3: "Hole"
}

# ===================== MÀU SẮC CHO MỖI LỚP =====================
CLASS_COLORS = {
    0: "#FFD700",  # Vàng - Scratch
    1: "#FF0000",  # Đỏ - Crack
    2: "#0000FF",  # Xanh dương - Bubble
    3: "#00FFFF"   # Xanh ngọc - Hole
}


def calculate_bbox_areas():
    """
    Tính diện tích của tất cả bounding boxes từ file label YOLO.
    YOLO format:
        class_id x_center y_center width height

    Diện tích bbox:
        area = width × height

    Do width và height đã normalized nên area nằm trong khoảng 0 đến 1.
    """
    bbox_areas = []
    class_areas = {class_id: [] for class_id in CLASS_NAMES.keys()}

    if not os.path.exists(LABELS_DIR):
        print(f"Thư mục không tồn tại: {LABELS_DIR}")
        return None, None

    label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")]
    print(f"Tìm thấy {len(label_files)} file labels")

    for label_file in label_files:
        label_path = os.path.join(LABELS_DIR, label_file)

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    continue

                if class_id not in class_areas:
                    continue

                area = width * height

                bbox_areas.append(area)
                class_areas[class_id].append(area)

    bbox_areas = np.array(bbox_areas)

    print(f"Tổng số bounding boxes: {len(bbox_areas)}")

    for class_id in CLASS_NAMES.keys():
        count = len(class_areas[class_id])
        if count > 0:
            avg_area = np.mean(class_areas[class_id])
            print(
                f"  {CLASS_NAMES[class_id]}: {count} boxes, "
                f"trung bình diện tích: {avg_area:.6f}"
            )
        else:
            print(f"  {CLASS_NAMES[class_id]}: 0 boxes")

    return bbox_areas, class_areas


def plot_histogram(bbox_areas, class_areas):
    """
    Vẽ histogram diện tích bounding boxes:
    - Biểu đồ 1: toàn bộ bounding boxes
    - Biểu đồ 2: phân bố theo từng lớp, dùng dạng đường viền để tránh che màu
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ===================== SUBPLOT 1: HISTOGRAM TOÀN BỘ =====================
    ax1 = axes[0]

    ax1.hist(
        bbox_areas,
        bins=50,
        edgecolor="black",
        alpha=0.7,
        color="skyblue"
    )

    ax1.set_xlabel("Diện tích Bounding Box (width × height)", fontsize=12)
    ax1.set_ylabel("Số lượng", fontsize=12)
    ax1.set_title(
        "Histogram Diện tích Bounding Boxes (Toàn bộ)",
        fontsize=14,
        fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    stats_text = (
        f"Tổng: {len(bbox_areas)}\n"
        f"TB: {np.mean(bbox_areas):.4f}\n"
        f"Min: {np.min(bbox_areas):.4f}\n"
        f"Max: {np.max(bbox_areas):.4f}"
    )

    ax1.text(
        0.98,
        0.97,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=10
    )

    # ===================== SUBPLOT 2: HISTOGRAM THEO LỚP =====================
    ax2 = axes[1]

    bins = np.linspace(np.min(bbox_areas), np.max(bbox_areas), 31)

    for class_id in CLASS_NAMES.keys():
        areas = class_areas[class_id]

        if len(areas) > 0:
            ax2.hist(
                areas,
                bins=bins,
                histtype="step",
                linewidth=2.4,
                label=f"{CLASS_NAMES[class_id]} (n={len(areas)})",
                color=CLASS_COLORS[class_id]
            )

    ax2.set_xlabel("Diện tích Bounding Box (width × height)", fontsize=12)
    ax2.set_ylabel("Số lượng", fontsize=12)
    ax2.set_title(
        "Histogram Diện tích Bounding Boxes (Theo lớp)",
        fontsize=14,
        fontweight="bold"
    )
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Dùng log scale để các lớp ít dữ liệu như Scratch/Crack dễ nhìn hơn
    ax2.set_yscale("log")

    plt.tight_layout()

    output_path = "bbox_area_histogram_fixed.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nHình đã lưu tại: {output_path}")

    plt.show()


def print_detailed_statistics(bbox_areas, class_areas):
    """
    In thống kê chi tiết diện tích bounding boxes.
    """
    print("\n" + "=" * 60)
    print("THỐNG KÊ CHI TIẾT DIỆN TÍCH BOUNDING BOXES")
    print("=" * 60)

    print("\nToàn bộ dataset:")
    print(f"  Tổng số boxes: {len(bbox_areas)}")
    print(f"  Trung bình: {np.mean(bbox_areas):.6f}")
    print(f"  Độ lệch chuẩn: {np.std(bbox_areas):.6f}")
    print(f"  Min: {np.min(bbox_areas):.6f}")
    print(f"  Max: {np.max(bbox_areas):.6f}")
    print(f"  Median (50%): {np.median(bbox_areas):.6f}")
    print(f"  25% percentile: {np.percentile(bbox_areas, 25):.6f}")
    print(f"  75% percentile: {np.percentile(bbox_areas, 75):.6f}")

    print("\nTheo lớp:")
    for class_id in CLASS_NAMES.keys():
        areas = np.array(class_areas[class_id])

        print(f"\n  {CLASS_NAMES[class_id]}:")

        if len(areas) > 0:
            print(f"    Số boxes: {len(areas)}")
            print(f"    Trung bình: {np.mean(areas):.6f}")
            print(f"    Độ lệch chuẩn: {np.std(areas):.6f}")
            print(f"    Min: {np.min(areas):.6f}")
            print(f"    Max: {np.max(areas):.6f}")
            print(f"    Median: {np.median(areas):.6f}")
            print(f"    25% percentile: {np.percentile(areas, 25):.6f}")
            print(f"    75% percentile: {np.percentile(areas, 75):.6f}")
        else:
            print("    Không có bounding box")


if __name__ == "__main__":
    bbox_areas, class_areas = calculate_bbox_areas()

    if bbox_areas is not None and len(bbox_areas) > 0:
        print_detailed_statistics(bbox_areas, class_areas)
        plot_histogram(bbox_areas, class_areas)
    else:
        print("Không thể tính diện tích bounding boxes!")