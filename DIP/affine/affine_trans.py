import cv2
import numpy as np

def put_label(image, text):
    output = image.copy()
    cv2.putText(output, text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2, cv2.LINE_AA)
    return output

# Read image
img = cv2.imread("C:/Users/calmnight/python3/10402AI_course/DIP/affine/bird2.jpg")
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

# Notice this is different from PIL library
h, w = img.shape[:2]

# 1. Identity transformation
M_identity = np.float32([[1, 0, 0], [0, 1, 0]])
img_identity = cv2.warpAffine(img, M_identity, (w, h))

# 2. Scaling
# x-direction: 1.2 times
# y-direction: 0.8 times
M_scaling = np.float32([[1.2, 0, 0], [0, 0.8, 0]])
img_scaling = cv2.warpAffine(img, M_scaling, (w, h))

# 3. Rotation
# Rotate 30 degrees around the image center, with scale=1.0
M_rotation = cv2.getRotationMatrix2D((w // 2, h // 2), 30, 1.0)
img_rotation = cv2.warpAffine(img, M_rotation, (w, h))

# 4. Translation
# Move right by 80 pixels and down by 50 pixels
M_translation = np.float32([[1, 0, 80], [0, 1, 50]])
img_translation = cv2.warpAffine(img, M_translation, (w, h))

# 5. Horizontal shear
# x' = x + 0.3*y
# y' = y
M_shear_h = np.float32([[1, 0.3, 0], [0, 1, 0]])
img_shear_h = cv2.warpAffine(img, M_shear_h, (w, h))

# 6. Vertical shear
# x' = x
# y' = y + 0.3*x
M_shear_v = np.float32([[1, 0, 0], [0.3, 1, 0]])
img_shear_v = cv2.warpAffine(img, M_shear_v, (w, h))

# Add labels
images = [
    put_label(img, "Original"),
    put_label(img_identity, "Identity"),
    put_label(img_scaling, "Scaling"),
    put_label(img_rotation, "Rotation"),
    put_label(img_translation, "Translation"),
    put_label(img_shear_h, "Shear Horizontal"),
    put_label(img_shear_v, "Shear Vertical"),
]

# Make all images same size
canvas1 = np.hstack(images[:3])
canvas2 = np.hstack(images[3:6])

# For the last image, pad with blank images to align
blank = np.zeros_like(img)
canvas3 = np.hstack([images[6], blank, blank])

result = np.vstack([canvas1, canvas2, canvas3])

cv2.imwrite("C:/Users/calmnight/python3/10402AI_course/DIP/affine/affine_demo.jpg", result)
cv2.imshow("Affine Transformation Demo", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
