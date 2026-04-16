import cv2
import numpy as np


# ---------------------------------
# Read the input image
# ---------------------------------
img = cv2.imread("C:/Users/calmnight/python3/10402AI_course/DIP/filter/Lena.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")


# ---------------------------------
# Define a low_pass filter mask
# 3x3 low_pass kernel:
# [1, 2, 1]
# [2, 4, 2]
# [1, 2, 1]
# Divide by 9 to get the average
# ---------------------------------
kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16.0


# ---------------------------------
# Apply low_pass filtering
# ---------------------------------
low_pass_filtered = cv2.filter2D(img, -1, kernel)


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
low_pass_filtered_labeled = add_label(low_pass_filtered, "low_pass Filtered Image")


# ---------------------------------
# Convert grayscale images to 3-channel for easier canvas stacking
# ---------------------------------
def to_3ch(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


panel1 = to_3ch(original_labeled)
panel2 = to_3ch(low_pass_filtered_labeled)

canvas = np.hstack([panel1, panel2])


# ---------------------------------
# Save and display results
# ---------------------------------
cv2.imwrite("C:/Users/calmnight/python3/10402AI_course/DIP/filter/low_pass_filter_demo.jpg", canvas)
cv2.imshow("low_pass Filter Demo", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done. Output saved as low_pass_filter_demo.jpg")
