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


def calculate_aspect_ratios():
    """
    Tính aspect ratio của tất cả bounding boxes từ file label YOLO.

    YOLO format:
        class_id x_center y_center width height

    Aspect ratio:
        AR = width / height
    """
    aspect_ratios = []
    class_ratios = {class_id: [] for class_id in CLASS_NAMES.keys()}

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

                if class_id not in class_ratios:
                    continue

                if height <= 0:
                    continue

                aspect_ratio = width / height

                aspect_ratios.append(aspect_ratio)
                class_ratios[class_id].append(aspect_ratio)

    aspect_ratios = np.array(aspect_ratios)

    print(f"Tổng số bounding boxes: {len(aspect_ratios)}")

    for class_id in CLASS_NAMES.keys():
        count = len(class_ratios[class_id])

        if count > 0:
            avg_ratio = np.mean(class_ratios[class_id])
            print(
                f"  {CLASS_NAMES[class_id]}: {count} boxes, "
                f"trung bình aspect ratio: {avg_ratio:.6f}"
            )
        else:
            print(f"  {CLASS_NAMES[class_id]}: 0 boxes")

    return aspect_ratios, class_ratios


def plot_histogram(aspect_ratios, class_ratios):
    """
    Vẽ histogram aspect ratio của bounding boxes:
    - Biểu đồ 1: toàn bộ bounding boxes
    - Biểu đồ 2: theo từng lớp, dùng đường viền để tránh che màu
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ===================== SUBPLOT 1: HISTOGRAM TOÀN BỘ =====================
    ax1 = axes[0]

    ax1.hist(
        aspect_ratios,
        bins=50,
        edgecolor="black",
        alpha=0.7,
        color="lightcoral"
    )

    ax1.set_xlabel("Aspect Ratio (width / height)", fontsize=12)
    ax1.set_ylabel("Số lượng", fontsize=12)
    ax1.set_title(
        "Histogram Aspect Ratio Bounding Boxes (Toàn bộ)",
        fontsize=14,
        fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    ax1.axvline(
        x=1.0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Perfect Square (AR=1.0)"
    )

    stats_text = (
        f"Tổng: {len(aspect_ratios)}\n"
        f"TB: {np.mean(aspect_ratios):.4f}\n"
        f"Min: {np.min(aspect_ratios):.4f}\n"
        f"Max: {np.max(aspect_ratios):.4f}"
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

    ax1.legend(fontsize=10)

    # ===================== SUBPLOT 2: HISTOGRAM THEO LỚP =====================
    ax2 = axes[1]

    bins = np.linspace(np.min(aspect_ratios), np.max(aspect_ratios), 31)

    for class_id in CLASS_NAMES.keys():
        ratios = class_ratios[class_id]

        if len(ratios) > 0:
            ax2.hist(
                ratios,
                bins=bins,
                histtype="step",
                linewidth=2.4,
                label=f"{CLASS_NAMES[class_id]} (n={len(ratios)})",
                color=CLASS_COLORS[class_id]
            )

    ax2.axvline(
        x=1.0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Perfect Square (AR=1.0)"
    )

    ax2.set_xlabel("Aspect Ratio (width / height)", fontsize=12)
    ax2.set_ylabel("Số lượng", fontsize=12)
    ax2.set_title(
        "Histogram Aspect Ratio Bounding Boxes (Theo lớp)",
        fontsize=14,
        fontweight="bold"
    )

    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")
    ax2.legend(fontsize=11)

    plt.tight_layout()

    output_path = "bbox_aspect_ratio_histogram_fixed.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nHình đã lưu tại: {output_path}")

    plt.show()


def print_detailed_statistics(aspect_ratios, class_ratios):
    """
    In ra các thống kê chi tiết aspect ratio bounding boxes.
    """
    print("\n" + "=" * 60)
    print("THỐNG KÊ CHI TIẾT ASPECT RATIO BOUNDING BOXES")
    print("=" * 60)

    print("\nToàn bộ dataset:")
    print(f"  Tổng số boxes: {len(aspect_ratios)}")
    print(f"  Trung bình: {np.mean(aspect_ratios):.6f}")
    print(f"  Độ lệch chuẩn: {np.std(aspect_ratios):.6f}")
    print(f"  Min: {np.min(aspect_ratios):.6f}")
    print(f"  Max: {np.max(aspect_ratios):.6f}")
    print(f"  Median (50%): {np.median(aspect_ratios):.6f}")
    print(f"  25% percentile: {np.percentile(aspect_ratios, 25):.6f}")
    print(f"  75% percentile: {np.percentile(aspect_ratios, 75):.6f}")

    square_count = np.sum((aspect_ratios >= 0.8) & (aspect_ratios <= 1.2))
    square_percent = square_count / len(aspect_ratios) * 100

    print(
        f"  Boxes gần hình vuông (0.8 <= AR <= 1.2): "
        f"{square_count} ({square_percent:.2f}%)"
    )

    print("\nTheo lớp:")
    for class_id in CLASS_NAMES.keys():
        ratios = np.array(class_ratios[class_id])

        print(f"\n  {CLASS_NAMES[class_id]}:")

        if len(ratios) > 0:
            print(f"    Số boxes: {len(ratios)}")
            print(f"    Trung bình: {np.mean(ratios):.6f}")
            print(f"    Độ lệch chuẩn: {np.std(ratios):.6f}")
            print(f"    Min: {np.min(ratios):.6f}")
            print(f"    Max: {np.max(ratios):.6f}")
            print(f"    Median: {np.median(ratios):.6f}")
            print(f"    25% percentile: {np.percentile(ratios, 25):.6f}")
            print(f"    75% percentile: {np.percentile(ratios, 75):.6f}")

            square_count_class = np.sum((ratios >= 0.8) & (ratios <= 1.2))
            square_percent_class = square_count_class / len(ratios) * 100

            print(
                f"    Boxes gần hình vuông: "
                f"{square_count_class} ({square_percent_class:.2f}%)"
            )
        else:
            print("    Không có bounding box")


if __name__ == "__main__":
    aspect_ratios, class_ratios = calculate_aspect_ratios()

    if aspect_ratios is not None and len(aspect_ratios) > 0:
        print_detailed_statistics(aspect_ratios, class_ratios)
        plot_histogram(aspect_ratios, class_ratios)
    else:
        print("Không thể tính aspect ratio của bounding boxes!")