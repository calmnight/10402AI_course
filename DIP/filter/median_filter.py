import cv2
import numpy as np


# ------------------------------
# 1. Read RGB image using OpenCV
# ------------------------------
img_bgr = cv2.imread("C:/Users/calmnight/python3/10402AI_course/DIP/filter/Lena.jpg")
if img_bgr is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

# OpenCV reads images in BGR order, so convert to RGB first
img_rgb = img_bgr[:, :, ::-1].copy()


# -------------------------------------------------
# 2. Convert RGB image to grayscale using NTSC rule
#    Gray = 0.299R + 0.587G + 0.114B
#    No OpenCV is used in this computation
# -------------------------------------------------
def rgb_to_grayscale_ntsc(rgb_image):
    rgb_float = rgb_image.astype(np.float32)
    r = rgb_float[:, :, 0]
    g = rgb_float[:, :, 1]
    b = rgb_float[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


# ---------------------------------
# 3. Add salt-and-pepper noise
#    No OpenCV is used here
# ---------------------------------
def add_salt_pepper_noise(gray_image, noise_ratio=0.05):
    noisy = gray_image.copy()
    h, w = gray_image.shape
    total_pixels = h * w
    num_noisy = int(total_pixels * noise_ratio)

    # Randomly choose pixel positions
    coords = np.random.choice(total_pixels, size=num_noisy, replace=False)
    ys = coords // w
    xs = coords % w

    # Randomly assign half to salt (255) and half to pepper (0)
    salt_mask = np.random.rand(num_noisy) > 0.5
    noisy[ys[salt_mask], xs[salt_mask]] = 255
    noisy[ys[~salt_mask], xs[~salt_mask]] = 0

    return noisy


# ---------------------------------
# 4. Median filter implementation
#    No OpenCV is used here
# ---------------------------------
def median_filter(gray_image, ksize=3):
    if ksize % 2 == 0:
        raise ValueError("ksize must be an odd number.")

    pad = ksize // 2
    padded = np.pad(gray_image, pad_width=pad, mode="edge")
    h, w = gray_image.shape
    filtered = np.zeros_like(gray_image)

    for y in range(h):
        for x in range(w):
            window = padded[y:y + ksize, x:x + ksize]
            filtered[y, x] = np.median(window)

    return filtered.astype(np.uint8)


# ------------------------------
# 5. Run the full image pipeline
# ------------------------------
gray_img = rgb_to_grayscale_ntsc(img_rgb)
noisy_img = add_salt_pepper_noise(gray_img, noise_ratio=0.05)
filtered_img = median_filter(noisy_img, ksize=3)


# ------------------------------------------------
# 6. Visualize all stages in one canvas using OpenCV
# ------------------------------------------------
def to_3ch(gray_image):
    return np.stack([gray_image, gray_image, gray_image], axis=2)


def add_label(image, text):
    output = image.copy()
    cv2.putText(
        output,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output


# Convert images to BGR for OpenCV display/saving
original_bgr = img_rgb[:, :, ::-1].copy()
gray_bgr = to_3ch(gray_img)[:, :, ::-1].copy()
noisy_bgr = to_3ch(noisy_img)[:, :, ::-1].copy()
filtered_bgr = to_3ch(filtered_img)[:, :, ::-1].copy()

# Resize for a more compact display if needed
h, w = original_bgr.shape[:2]
display_w = 400
scale = display_w / w
display_h = int(h * scale)

def resize_display(image):
    return cv2.resize(image, (display_w, display_h))

panel1 = add_label(resize_display(original_bgr), "Original RGB")
panel2 = add_label(resize_display(gray_bgr), "Grayscale (NTSC)")
panel3 = add_label(resize_display(noisy_bgr), "Salt-and-Pepper Noise")
panel4 = add_label(resize_display(filtered_bgr), "Median Filter Result")

canvas_top = np.hstack([panel1, panel2])
canvas_bottom = np.hstack([panel3, panel4])
canvas = np.vstack([canvas_top, canvas_bottom])

cv2.imwrite("C:/Users/calmnight/python3/10402AI_course/DIP/filter/image_processing_pipeline.jpg", canvas)
cv2.imshow("RGB -> Grayscale -> Noise -> Median Filter", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done. Output saved as image_processing_pipeline.jpg")
