import cv2
import numpy as np


# ---------------------------------
# Read the input image
# ---------------------------------
img = cv2.imread("C:/Users/calmnight/python3/10402AI_course/DIP/filter/Lena.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")


# ---------------------------------
# Define a Laplacian mask
# Common 3x3 Laplacian kernel:
# [ 0, -1,  0]
# [-1,  4, -1]
# [ 0, -1,  0]
# ---------------------------------
kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)


# ---------------------------------
# Apply Laplacian filtering
# Use CV_32F to preserve negative values
# ---------------------------------
laplacian = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kernel)


# ---------------------------------
# Sharpen the image
# g(x, y) = f(x, y) - Laplacian(f)
# This form is used with the positive-center kernel above
# For visualization, sharpened values should be clip from 0 to 255
# ---------------------------------
sharpened = img.astype(np.float32) - laplacian
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------
# Normalize Laplacian result for display only because of negative values
# This step is only for visualization
# ---------------------------------
laplacian_display = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
laplacian_display = laplacian_display.astype(np.uint8)


# ---------------------------------
# Add labels to images
# ---------------------------------
def add_label(image, text):
    output = image.copy()
    cv2.putText(
        output,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        255,
        2,
        cv2.LINE_AA,
    )
    return output


original_labeled = add_label(img, "Original Image")
laplacian_labeled = add_label(laplacian_display, "Laplacian Response")
sharpened_labeled = add_label(sharpened, "Sharpened Image")


# ---------------------------------
# Convert grayscale images to 3-channel for easier canvas stacking (still grayscale-like)
# ---------------------------------
def to_3ch(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


panel1 = to_3ch(original_labeled)
panel2 = to_3ch(laplacian_labeled)
panel3 = to_3ch(sharpened_labeled)

canvas = np.hstack([panel1, panel2, panel3])


# ---------------------------------
# Save and display results
# ---------------------------------
cv2.imwrite("C:/Users/calmnight/python3/10402AI_course/DIP/filter/laplacian_sharpening_demo.jpg", canvas)
cv2.imshow("Laplacian Spatial Filter Sharpening", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done. Output saved as laplacian_sharpening_demo.jpg")
