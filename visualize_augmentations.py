import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================
# ĐƯỜNG DẪN ĐẾN 4 ẢNH GỐC DÙNG CHO MOSAIC
# =====================================================

img_paths = [
    r"dataset/images/train/originaldataset_000003.jpg",
    r"dataset/images/train/originaldataset_000004.jpg",
    r"dataset/images/train/originaldataset_000005.jpg",
    r"dataset/images/train/originaldataset_000008.jpg"
]

# =====================================================
# ĐỌC ẢNH
# =====================================================

images = []

for p in img_paths:
    if os.path.exists(p):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    else:
        print(f"Error: Could not find image at {p}")

if len(images) == 0:
    print("No images were loaded. Please check the paths.")
    exit(1)

original_img = images[0]

h, w, _ = original_img.shape

# =====================================================
# 1. BRIGHTNESS AUGMENTATION
# =====================================================

brightness_img = cv2.convertScaleAbs(
    original_img,
    alpha=1.2,
    beta=30
)

# =====================================================
# 2. GAUSSIAN NOISE
# =====================================================

gauss = np.random.normal(
    loc=0,
    scale=25,
    size=original_img.shape
)

noise_img = np.clip(
    original_img.astype(np.float32) + gauss,
    0,
    255
).astype(np.uint8)

# =====================================================
# 3. HORIZONTAL FLIP
# =====================================================

flip_img = cv2.flip(original_img, 1)

# =====================================================
# 4. ROTATION (45°)
# =====================================================

center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(
    center=center,
    angle=45,
    scale=1.0
)

rotated_img = cv2.warpAffine(
    original_img,
    M,
    (w, h)
)

# =====================================================
# 5. HSV AUGMENTATION
# =====================================================

hsv_img = cv2.cvtColor(
    original_img,
    cv2.COLOR_RGB2HSV
).astype(np.float32)

# Hue shift
hsv_img[:, :, 0] = (hsv_img[:, :, 0] + 30) % 180

# Saturation shift
hsv_img[:, :, 1] = np.clip(
    hsv_img[:, :, 1] * 1.5,
    0,
    255
)

hsv_img = hsv_img.astype(np.uint8)

hsv_transformed = cv2.cvtColor(
    hsv_img,
    cv2.COLOR_HSV2RGB
)

# =====================================================
# 6. MOSAIC AUGMENTATION
# =====================================================

if len(images) >= 4:

    mosaic_img = np.zeros(
        (h * 2, w * 2, 3),
        dtype=np.uint8
    )

    mosaic_img[0:h, 0:w] = cv2.resize(images[0], (w, h))
    mosaic_img[0:h, w:w*2] = cv2.resize(images[1], (w, h))
    mosaic_img[h:h*2, 0:w] = cv2.resize(images[2], (w, h))
    mosaic_img[h:h*2, w:w*2] = cv2.resize(images[3], (w, h))

    # Resize lại về kích thước ban đầu
    mosaic_img = cv2.resize(
        mosaic_img,
        (w, h)
    )

else:
    print("Need 4 images for Mosaic. Using original image.")
    mosaic_img = original_img

# =====================================================
# DANH SÁCH ẢNH HIỂN THỊ
# =====================================================

titles = [
    "(a) Original",
    "(b) Brightness",
    "(c) Gaussian Noise",
    "(d) Horizontal Flip",
    "(e) Rotation",
    "(f) HSV Transformation",
    "(g) Mosaic"
]

display_images = [
    original_img,
    brightness_img,
    noise_img,
    flip_img,
    rotated_img,
    hsv_transformed,
    mosaic_img
]

# =====================================================
# HIỂN THỊ THEO BỐ CỤC 2 HÀNG × 4 CỘT
# =====================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(16, 8)
)

axes = axes.flatten()

for i, ax in enumerate(axes):

    if i < len(display_images):

        ax.imshow(display_images[i])

        ax.set_title(
            titles[i],
            fontsize=11,
            fontweight='bold'
        )

    ax.axis("off")

# =====================================================
# CĂN CHỈNH KHOẢNG CÁCH
# =====================================================

plt.tight_layout()

# =====================================================
# LƯU ẢNH
# =====================================================

output_path = "augmentations_visualization.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Visualization saved to: {output_path}")

# =====================================================
# HIỂN THỊ
# =====================================================

plt.show()